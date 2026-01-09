/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {CompiledModel, loadAndCompile, loadLiteRt} from '@litertjs/core';
import {html, LitElement} from 'lit';
import {customElement, state} from 'lit/decorators.js';
// Placeholder for internal dependency on trusted resource url

import {MODEL_URL} from './constants';
import {DepthEstimatorOptions, runDepthEstimation} from './depth_estimator';
import {componentStyles} from './styles';


/* tslint:disable:no-new-decorators */

/**
 * A component that allows the user to select an image and generate a depth map
 * using LiteRT.
 */
@customElement('depth-anything')
export class DepthAnything extends LitElement {
  static override styles = componentStyles;

  @state() private statusMessage = 'Initializing LiteRT...';
  @state() private progressValue = 0;
  @state() private originalImage: HTMLImageElement|null = null;
  @state() private originalSrc = '';
  @state() private resultCanvas: HTMLCanvasElement|null = null;
  @state() private isProcessing = false;
  @state() private colormap: DepthEstimatorOptions['colormap'] = 'spectral_r';
  @state() private model: CompiledModel|null = null;
  @state() private isWebcamActive = false;
  private videoElement: HTMLVideoElement|null = null;

  override async firstUpdated() {
    try {
      await loadLiteRt('./wasm/', {threads: true});
      this.statusMessage = 'Ready. Please select an image.';
    } catch (e) {
      console.warn(
          'Failed to load LiteRT with threads: true, falling back to threads: false',
          e);
      this.statusMessage = `Retrying initialization without threading...`;
      try {
        await loadLiteRt('./wasm/', {threads: false});
        this.statusMessage = 'Ready. Please select an image.';
      } catch (e2) {
        this.statusMessage =
            `Error initializing LiteRT: ${(e2 as Error).message}`;
        console.error('Failed to load LiteRT with threads: false', e2);
        return;
      }
    }
    await this.loadModel();
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.stopWebcam();
  }

  private async loadModel() {
    if (this.model) return;

    this.statusMessage = `Downloading & compiling model...`;
    try {
      this.model = await loadAndCompile(MODEL_URL, {accelerator: 'wasm'});
      this.statusMessage = 'Ready. Please select an image.';
    } catch (e) {
      this.statusMessage = `Error loading model: ${(e as Error).message}`;
      console.error(e);
    }
  }

  private handleFileSelect(file: File) {
    if (!file.type.startsWith('image/')) {
      this.statusMessage = 'Please select an image file.';
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        this.originalImage = img;
        this.resultCanvas = null;
      };
      this.originalSrc = e.target?.result as string;
      img.src = this.originalSrc;
      this.statusMessage = 'Image loaded. Click "Run".';
    };
    reader.readAsDataURL(file);
  }

  private onDrop(e: DragEvent) {
    e.preventDefault();
    this.handleFileSelect(e.dataTransfer?.files[0] as File);
  }

  private onFileChange(e: Event) {
    const input = e.target as HTMLInputElement;
    this.handleFileSelect(input.files?.[0] as File);
  }

  private async handleDepthEstimation() {
    if (!this.originalImage || !this.model) {
      this.statusMessage =
          'Please load an image and wait for the model to compile.';
      return;
    }

    this.isProcessing = true;
    this.resultCanvas = null;
    try {
      const resultCanvas = await runDepthEstimation({
        sourceImage: this.originalImage,
        model: this.model,
        colormap: this.colormap,
        progressCallback: ({message, value}) => {
          this.statusMessage = message;
          this.progressValue = value;
        },
      });
      this.resultCanvas = resultCanvas;
      this.statusMessage = 'Depth estimation complete!';
    } catch (e) {
      this.statusMessage = `Error during processing: ${(e as Error).message}`;
      console.error(e);
    } finally {
      this.isProcessing = false;
    }
  }

  private onColormapChange(e: Event) {
    const colormap = (e.target as HTMLSelectElement).value;
    if (colormap === 'spectral_r') {
      this.colormap = 'spectral_r';
    } else if (colormap === 'grayscale') {
      this.colormap = 'grayscale';
    } else {
      console.error(`Unsupported colormap: ${colormap}`);
      this.colormap = 'spectral_r';
    }
  }

  private async handleWebcamClick() {
    if (this.isWebcamActive) {
      this.captureFrame();
      this.stopWebcam();
    } else {
      await this.startWebcam();
    }
  }

  private async startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      this.isWebcamActive = true;
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = stream;
      this.videoElement.autoplay = true;
      this.videoElement.style.maxWidth = '100%';
      this.videoElement.style.maxHeight = '70vh';
      this.videoElement.style.objectFit = 'contain';

      this.videoElement.onloadedmetadata = () => {
        this.requestUpdate();
      };

    } catch (e) {
      this.statusMessage = `Error accessing webcam: ${(e as Error).message}`;
      console.error(e);
    }
  }

  private stopWebcam() {
    if (this.videoElement &&
        this.videoElement.srcObject instanceof MediaStream) {
      const stream = this.videoElement.srcObject;
      stream.getTracks().forEach(track => track.stop());
    }
    this.isWebcamActive = false;
    this.videoElement = null;
    this.requestUpdate();
  }

  private captureFrame() {
    if (!this.videoElement) return;

    const canvas = document.createElement('canvas');
    canvas.width = this.videoElement.videoWidth;
    canvas.height = this.videoElement.videoHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

    const img = new Image();
    img.onload = () => {
      this.originalImage = img;
      this.resultCanvas = null;
      this.statusMessage = 'Image captured. Click "Run".';
    };
    this.originalSrc = canvas.toDataURL('image/png');
    img.src = this.originalSrc;
  }

  override render() {
    return html`
      <div class="container">
        <h1>🖼️ LiteRT.js Depth Anything</h1>
        <div class="controls">
          <div class="control-group">
            <label for="colormap-select">Colormap</label>
            <select
              id="colormap-select"
              @change=${this.onColormapChange}
              .value=${this.colormap}
            >
              <option value="spectral_r">Spectral_r</option>
              <option value="grayscale">Grayscale</option>
            </select>
          </div>
          <div class="control-group">
            <div class="license-info">
              <a href="https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE" target="_blank">
                Model License
              </a>
            </div>
            <div class="button-group">
              <button @click=${this.handleDepthEstimation} .disabled=${
    !this.originalImage || !this.model || this.isProcessing}>
                ${
        this.isProcessing ?
        'Processing...' :
        '🚀 Run'}
              </button>
              <button @click=${this.handleWebcamClick}>
                ${this.isWebcamActive ? 'Capture' : '📷 Webcam'}
              </button>
            </div>
          </div>
        </div>

        <div
          class="drop-zone"
          @dragover=${(e: DragEvent) => e.preventDefault()}
          @drop=${this.onDrop}
          @click=${
        () => {
          if (!this.isWebcamActive) {
            this.shadowRoot?.querySelector<HTMLInputElement>('#file-input')
                ?.click();
          }
        }}
        >
          ${
        this.isWebcamActive && this.videoElement ?
        html`
              <div class="webcam-container">
                ${this.videoElement}
              </div>
            ` :
        this.resultCanvas ?
        this.resultCanvas :
        this.originalSrc ?
        html`
            <img src=${this.originalSrc} alt="display image" />
          ` :
        html`
            <p>Drag & Drop an Image Here, or Click to Select</p>
          `}
          <input type="file" id="file-input" @change=${
        this.onFileChange} accept="image/*" hidden>
        </div>

        <div class="footer">
          <p class="status">${this.statusMessage}</p>
          ${
        this.isProcessing ?
        html`<progress max="1" .value=${this.progressValue}></progress>` :
        ''}
        </div>
      </div>
    `;
  }
}
