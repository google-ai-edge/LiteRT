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

import './console_mirror';
import './run_result_visualizer';
import './common_settings';

import {CompiledModel, SignatureRunner} from '@litertjs/core';
import {css, html, LitElement} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';
import {when} from 'lit/directives/when.js';

import type {Settings, SettingValues} from './common_settings';
import {formStyles, hostStyles, typographyStyles} from './common_styles';
import {ConsoleMirror} from './console_mirror';
import {LiteRtModelRunner} from './litert_model_runner';
import type {RunResult} from './model_runner';

/* tslint:disable:no-new-decorators */

/**
 * The main entrypoint and UI for the model tester.
 */
@customElement('model-tester')
export class ModelTester extends LitElement {
  static override styles = [
    hostStyles,
    typographyStyles,
    formStyles,
    css`
      :host {
        padding: 24px;
        max-width: 1024px;
        margin: 0 auto;
        background-color: #f9f9f9;
      }

      .header {
        padding-bottom: 16px;
        border-bottom: 1px solid #eee;
        margin-bottom: 16px;
      }

      .header h1 {
        margin: 0 0 8px 0;
      }

      .radio {
        margin-top: 12px;
      }

      .radio label {
        margin-left: 8px;
      }

      .file-input-container {
        margin: 16px 0;
      }

      .controls > * {
        margin-top: 16px;
      }
      .settings-and-signatures {
        display: flex;
        gap: 24px;
      }

      .divider {
        border-left: 1px solid #eee;
        margin: 0 12px;
      }
    `,
  ];

  @state() private modelRunner?: LiteRtModelRunner;
  @state() private running = false;
  @state() private runResult?: RunResult;
  @state() private selectedSignatureId?: string;

  @state()
  settingsDefinition: Settings = [
    {key: 'benchmark', label: 'Run Benchmark', dataType: 'boolean'},
    {key: 'benchmarkRuns', label: 'Number of Runs', dataType: 'number'},
  ];

  @state()
  settingsValues: SettingValues = [
    {key: 'benchmark', value: true},
    {key: 'benchmarkRuns', value: 10},
  ];

  @query('console-mirror') private consoleMirror!: ConsoleMirror;

  private getSetting(key: string) {
    return this.settingsValues.find(v => v.key === key)?.value;
  }

  private getBenchmarkCount() {
    // Only run benchmark if it is enabled.
    return this.getSetting('benchmark') ?
        (this.getSetting('benchmarkRuns') as number) :
        0;
  }

  private handleSettingsChange(event: CustomEvent) {
    const {key, value} = event.detail;
    this.settingsValues = this.settingsValues.map(
        setting => setting.key === key ? {...setting, value} : setting,
    );
  }

  private async setModelFile(event: InputEvent) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      const file = input.files[0];
      const data = new Uint8Array(await file.arrayBuffer());
      if (file.name.endsWith('.tflite')) {
        this.modelRunner = await LiteRtModelRunner.load(
            data,
            () => this.consoleMirror.getMessages(),
        );
      }
    }
  }

  private async runTest() {
    try {
      this.running = true;
      this.runResult = await this.modelRunner?.run(
          this.selectedSignatureId,
          this.getBenchmarkCount(),
      );
      return this.runResult;
    } finally {
      this.running = false;
    }
  }

  private renderSignatures() {
    const signatures = this.modelRunner?.getSignatures() ?? [];

    return html`
      <div>
        ${
        signatures.map(
            ({name, id, signature}) => html`
              <div class="radio">
                <input
                  type="radio"
                  id=${id}
                  name="signatures"
                  value=${id}
                  ?checked=${this.selectedSignatureId === id}
                  @click=${() => {
              this.selectedSignatureId = id;
            }}
                />
                <label for=${id}>${name}</label>
                ${
                when(
                    signature,
                    () => html`
                      <div class="code">${formatSignature(signature!)}</div>
                    `,
                    )}
              </div>
            `,
            )}
      </div>
    `;
  }

  override render() {
    return html`
      <div class="header">
        <h1>LiteRT Web Model Tester</h1>
        <p>
          This tool tests a LiteRT model running in WebGPU against a set of fake
          inputs to check if there are any WebGPU errors when loading or
          running the model.
        </p>
      </div>
    <div class="file-input-container">
      <label for="model-file-input" class="file-input-label">
        Choose Model File
      </label>
      <input
        id="model-file-input"
        type="file"
        @change=${this.setModelFile}
      />
    </div>

    <div class="settings-and-signatures">
      <common-settings
        .settings=${this.settingsDefinition}
        .values=${this.settingsValues}
        @settings-changed=${this.handleSettingsChange}
      ></common-settings>
      <div class="divider"></div>
      <div><h4>Signatures</h4>
      ${when(this.modelRunner, () => this.renderSignatures())}
      </div>
    </div>

    <div class="controls">
      <button
        class="primary"
        ?disabled=${!this.modelRunner || this.running}
        @click=${() => {
      this.runTest();
    }}>
        Run
      </button>
    </div>
    ${
        when(
            this.runResult,
            () => html`
              <div>
                <run-result-visualizer
                  .runResult=${this.runResult}
                ></run-result-visualizer>
              </div>
            `,
            )}
    <console-mirror></console-mirror>
    `;
  }
}

function formatSignature(signature: SignatureRunner|CompiledModel) {
  const inputs = signature.getInputDetails();
  const inputsString = inputs.map(input => formatTensorData(input)).join(', ');

  const outputs = signature.getOutputDetails();
  const outputsString =
      outputs.map(output => formatTensorData(output)).join(', ');

  return `(${inputsString}) => [${outputsString}]`;
}

function formatTensorData(data: {shape: Int32Array; dtype: string}) {
  return Object.values(data.shape).join('x') + 'x' + data.dtype;
}
