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

import {CompiledModel, isWebGPUSupported, loadAndCompile, loadLiteRt} from '@litertjs/core';
import {html, LitElement, TemplateResult} from 'lit';
import {customElement, state} from 'lit/decorators.js';
// Placeholder for internal dependency on trusted resource url

import {componentStyles} from './styles';
import {upscaleImageWithTiling} from './upscaler';


interface ModelInfo {
  url: string;
  // The expected normalization range for the model's float32 input
  range: [number, number];
  licenseHtml: TemplateResult;
}

const MODELS: Record<string, ModelInfo> = {
  'Real-ESRGAN x4plus': {
    url:
        'https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/v0.37.0/Real-ESRGAN-x4plus_float.tflite',
    licenseHtml: html`
      <div class="license-info">
        <a href="https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE" target="_blank">Model License</a>
        |
        <a href="https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/blob/main/DEPLOYMENT_MODEL_LICENSE.pdf" target="_blank">Deployment License</a>
      </div>
    `,
    range: [0, 1],  // Normalizes to [0, 1]
  },
};

/* tslint:disable:no-new-decorators */

/**
 * A component that allows the user to select an image and upscale it using
 * LiteRT.
 */
@customElement('image-upscaler')
export class ImageUpscaler extends LitElement {
  static override styles = componentStyles;

  @state() private statusMessage = 'Initializing LiteRT...';
  @state() private progressValue = 0;
  @state() private originalImage: HTMLImageElement|null = null;
  @state() private originalSrc = '';
  @state() private upscaledCanvas: HTMLCanvasElement|null = null;
  @state() private isUpscaling = false;
  @state() private sliderValue = 50;
  @state() private isDraggingSlider = false;
  @state() private preventClick = false;
  private comparisonContainerRect: DOMRect|null = null;
  private dragStartX: number|null = null;

  @state() private models: Record<string, CompiledModel|null> = {};
  @state() private selectedModelName: string = Object.keys(MODELS)[0];
  @state() private overlapPercent = 20;

  override async firstUpdated() {
    try {
      await loadLiteRt('./wasm/');
      this.statusMessage = 'Ready. Please select an image.';
      await this.loadModel(this.selectedModelName);
    } catch (e) {
      this.statusMessage = `Error initializing LiteRT: ${(e as Error).message}`;
      console.error(e);
    }
  }

  private async loadModel(name: string) {
    if (this.models[name]) return;

    this.models = {...this.models, [name]: null};
    this.statusMessage = `Downloading & compiling ${name}...`;
    try {
      const accelerator = isWebGPUSupported() ? 'webgpu' : 'wasm';
      // UPDATE: Use modelInfo.url to get the path
      const modelInfo = MODELS[name];
      const model = await loadAndCompile(modelInfo.url, {accelerator});
      this.models = {...this.models, [name]: model};
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
        this.upscaledCanvas = null;
      };
      this.originalSrc = e.target?.result as string;
      img.src = this.originalSrc;
      this.statusMessage = 'Image loaded. Click "Upscale".';
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

  private onModelChange(e: Event) {
    this.selectedModelName = (e.target as HTMLSelectElement).value;
    this.loadModel(this.selectedModelName);
  }

  private readonly handleDragMove = (e: MouseEvent) => {
    if (!this.isDraggingSlider || !this.comparisonContainerRect) {
      return;
    }

    // If the mouse button was released outside the window, stop dragging.
    if (e.buttons === 0) {
      this.stopDrag();
      return;
    }

    // If the mouse has moved more than a few pixels, treat it as a drag
    // and prevent the click event from firing.
    if (this.dragStartX !== null && Math.abs(e.clientX - this.dragStartX) > 5) {
      this.preventClick = true;
    }

    const x = e.clientX - this.comparisonContainerRect.left;
    const percent = (x / this.comparisonContainerRect.width) * 100;
    this.sliderValue = Math.max(0, Math.min(100, percent));
  };

  private readonly stopDrag = () => {
    this.isDraggingSlider = false;
    this.comparisonContainerRect = null;
    this.dragStartX = null;

    window.removeEventListener('mousemove', this.handleDragMove);
    window.removeEventListener('mouseup', this.stopDrag);

    // Use a timeout to distinguish a drag from a click. If we just dragged,
    // we don't want the click event on the drop-zone to fire.
    setTimeout(() => {
      this.preventClick = false;
    }, 0);
  };

  private readonly startDrag = (e: MouseEvent) => {
    // Only drag with the main mouse button
    if (e.button !== 0) {
      return;
    }
    e.preventDefault();  // Prevent text selection/image dragging

    this.isDraggingSlider = true;
    // Assume it's a click until the mouse moves significantly
    this.preventClick = false;
    this.dragStartX = e.clientX;
    const container = e.currentTarget as HTMLElement;
    this.comparisonContainerRect = container.getBoundingClientRect();

    // Add global listeners to track mouse movement outside the container
    window.addEventListener('mousemove', this.handleDragMove);
    window.addEventListener('mouseup', this.stopDrag);
  };


  private renderComparison() {
    return html`
      <div class="comparison-container"
           @mousedown=${this.startDrag}
      >
        <img
          class="comparison-img"
          src=${this.originalSrc}
          draggable="false"
        />
        <div class="comparison-img-wrapper" style="clip-path: inset(0 0 0 ${
        this.sliderValue}%)">
          ${this.upscaledCanvas}
        </div>
        <div class="comparison-slider-bar" style="left: ${
        this.sliderValue}%"></div>
      </div>
    `;
  }

  private async handleUpscale() {
    const model = this.models[this.selectedModelName];
    // Get the normalization range from our model list
    const modelInfo = MODELS[this.selectedModelName];

    if (!this.originalImage || !model) {
      this.statusMessage =
          'Please load an image and wait for the model to compile.';
      return;
    }

    this.isUpscaling = true;
    this.upscaledCanvas = null;
    try {
      const resultCanvas = await upscaleImageWithTiling({
        sourceImage: this.originalImage,
        model,
        overlapPercent: this.overlapPercent,
        normalizationRange: modelInfo.range,
        progressCallback: ({message, value}) => {
          this.statusMessage = message;
          this.progressValue = value;
        },
      });
      this.upscaledCanvas = resultCanvas;
      this.statusMessage = 'Upscaling complete!';
    } catch (e) {
      this.statusMessage = `Error during upscaling: ${(e as Error).message}`;
      console.error(e);
    } finally {
      this.isUpscaling = false;
    }
  }

  override render() {
    const currentModel = this.models[this.selectedModelName];

    return html`
      <div class="container">
        <h1>🖼️ LiteRT.js Image Upscaler</h1>
        <div class="controls">
          <div class="control-group">
            <label for="model-select">Model:</label>
            <select id="model-select" @change=${this.onModelChange}>
              ${
        Object.keys(MODELS).map(
            name => html`<option .value=${name}>${name}</option>`)}
            </select>
            ${MODELS[this.selectedModelName]?.licenseHtml ?? ''}
          </div>
          <div class="control-group">
            <label for="overlap-slider">Tile Overlap: ${
        this.overlapPercent}%</label>
            <input
              type="range"
              id="overlap-slider"
              min="0"
              max="50"
              .value=${`${this.overlapPercent}`}
              @input=${
        (e: Event) => this.overlapPercent =
            Number((e.target as HTMLInputElement).value)}>
          </div>
          <button @click=${this.handleUpscale} .disabled=${
    !this.originalImage || !currentModel || this.isUpscaling}>
            ${
        this.isUpscaling ?
        'Upscaling...' :
        '🚀 Upscale'}
          </button>
        </div>

        <div
          class="drop-zone"
          @dragover=${(e: DragEvent) => e.preventDefault()}
          @drop=${this.onDrop}
          @click=${
        () => {
          if (this.preventClick) return;
          this.shadowRoot?.querySelector<HTMLInputElement>('#file-input')
            ?.click();
        }}
        >
          ${
        this.upscaledCanvas && this.originalSrc ?
        this.renderComparison() :
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
        this.isUpscaling ?
        html`<progress max="1" .value=${this.progressValue}></progress>` :
        ''}
        </div>
      </div>
    `;
  }
}
