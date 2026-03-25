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
import '@material/web/tabs/tabs';
import '@material/web/tabs/primary-tab';

import {CompiledModel, SignatureRunner} from '@litertjs/core';
import {MdTabs} from '@material/web/tabs/tabs';
import {css, html, LitElement} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';
import {choose} from 'lit/directives/choose.js';
import {when} from 'lit/directives/when.js';

import type {Settings, SettingValues} from './common_settings';
import {formStyles, hostStyles, typographyStyles} from './common_styles';
import {ConsoleMirror} from './console_mirror';
import {download} from './download';
import {getFileHandle} from './file_utils';
import {LiteRtModelRunner} from './litert_model_runner';
import {locationReplace} from './location_replace';
import type {RunResult} from './model_runner';
import {determineStatus} from './run_result_list';
import {clearTestResults, loadTestState, saveTestState, selectDirectory, type TestState} from './test_state';

const URL_PARAMS = {
  runTest: 'runTest',
  testResults: 'testResults',
};

const MODEL_TIMEOUT = 60_000;

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
  @state() private testState?: TestState;
  @state() private results?: TestState['results'];

  @state()
  settingsDefinition: Settings = [
    {key: 'benchmark', label: 'Run Benchmark', dataType: 'boolean'},
    {key: 'benchmarkRuns', label: 'Number of Runs', dataType: 'number'},
    {key: 'warmupRuns', label: 'Number of Warmup Runs', dataType: 'number'},
    {
      key: 'convertTensorsPerRun',
      label: 'Convert Tensors Per Run',
      dataType: 'boolean'
    },
  ];

  @state()
  settingsValues: SettingValues = [
    {key: 'benchmark', value: true},
    {key: 'benchmarkRuns', value: 10},
    {key: 'warmupRuns', value: 3},
    {key: 'convertTensorsPerRun', value: true},
  ];
  @state() private activeTabIndex = 0;

  @query('console-mirror') private consoleMirror!: ConsoleMirror;
  @query('md-tabs') private tabs!: MdTabs;

  private getSetting(key: string) {
    return this.settingsValues.find(v => v.key === key)?.value;
  }

  private getBenchmarkCount() {
    // Only run benchmark if it is enabled.
    return this.getSetting('benchmark') ?
        (this.getSetting('benchmarkRuns') as number) :
        0;
  }

  private getWarmupCount() {
    return this.getSetting('benchmark') ?
        (this.getSetting('warmupRuns') as number) :
        0;
  }

  private handleSettingsChange(event: CustomEvent) {
    const {key, value} = event.detail;
    this.settingsValues = this.settingsValues.map(
        setting => setting.key === key ? {...setting, value} : setting,
    );
  }

  private async runTests() {
    this.testState = await clearTestResults();
    if (!this.testState) {
      throw new Error('Tests not configured. Please select files to test');
    }

    this.results = this.testState.results;
    this.testState.settings = this.settingsValues;

    // Save the set of tests to run once we reload the page with the `runTest`
    // URL param.
    await saveTestState(this.testState);

    const url = new URL(window.location.href);

    url.searchParams.set(URL_PARAMS.runTest, 'true');
    url.searchParams.delete(URL_PARAMS.testResults);
    locationReplace(url);
  }

  override async firstUpdated() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get(URL_PARAMS.runTest) === 'true') {
      this.testState = await loadTestState();
      this.settingsValues = this.testState?.settings ?? this.settingsValues;
      this.results = this.testState?.results;

      this.activeTabIndex = 1;
      this.tabs!.activeTabIndex = 1;

      if (this.testState &&
          this.testState.pathIndex < this.testState.pathsToTest.length) {
        try {
          const filePath = this.testState.pathsToTest[this.testState.pathIndex];
          const handle = await getFileHandle(
              this.testState.filesystemHandle,
              filePath,
          );
          const file = await handle.getFile();
          const data = new Uint8Array(await file.arrayBuffer());

          if (filePath.endsWith('.tflite')) {
            this.modelRunner = await Promise.race([
              LiteRtModelRunner.load(
                  data,
                  () => this.consoleMirror.getMessages(),
                  ),
              deadline(
                  MODEL_TIMEOUT,
                  new Error(`Failed to load model in ${MODEL_TIMEOUT / 1000}s`),
                  ),
            ]);
          } else {
            throw new Error(`Unknown file type ${filePath}`);
          }

          this.testState.results[filePath] = removeTensorData(
              await this.modelRunner.run(
                  undefined,
                  this.getBenchmarkCount(),
                  this.getWarmupCount(),
                  this.getSetting('convertTensorsPerRun') as boolean,
                  ),
          );

          this.requestUpdate();
        } catch (e) {
          console.error('Failed to run test', e);
        }

        this.testState.pathIndex++;
        await saveTestState(this.testState);
        if (this.testState.pathIndex < this.testState.pathsToTest.length) {
          // Run the next test
          window.location.reload();
        } else {
          const url = new URL(window.location.href);
          url.searchParams.delete(URL_PARAMS.runTest);
          window.history.replaceState({}, '', url);
        }
      }
    }

    if (urlParams.has(URL_PARAMS.testResults)) {
      try {
        const resultsPath = urlParams.get(URL_PARAMS.testResults)!;
        const parsed = await (await fetch(resultsPath)).json();
        this.results = parsed['results'];
        this.activeTabIndex = 1;
        this.tabs!.activeTabIndex = 1;
      } catch (e) {
        console.error(e);
      }
    }
  }

  private async setModelFile(event: Event) {
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
          this.getWarmupCount(),
          this.getSetting('convertTensorsPerRun') as boolean,
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
                  id=${id ?? 'base_signature'}
                  name="signatures"
                  value=${id ?? 'base_signature'}
                  ?checked=${this.selectedSignatureId === id}
                  @click=${() => {
              this.selectedSignatureId = id;
            }}
                />
                <label for=${id ?? 'base_signature'}>${name}</label>
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

  private renderSingleModelTester() {
    return html`
    <div class="file-input-container">
      <label for="model-file-input" class="file-input-label">
        Choose Model File
      </label>
      <input
        id="model-file-input"
        type="file"
        @change=${this.setModelFile}
      />
      <button
        class="primary"
        ?disabled=${!this.modelRunner || this.running}
        @click=${() => {
      this.runTest();
    }}>
        Run
      </button>
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
    `;
  }

  private renderBatchedModelTester() {
    return html`
      <div class="controls">
        <button
          @click=${async () => {
      this.testState = {
        settings: this.settingsValues,
        ...await selectDirectory(),
      };
      this.results = this.testState.results;
    }}
        >
          Choose Directory
        </button>
        <button
          class="primary"
          @click=${() => this.runTests()}
          ?disabled=${!this.testState}
        >
          Run tests
        </button>
        <button @click=${() => this.downloadBatchedResults()}>
          Download Results
        </button>
      </div>

      <div class="file-input-container">
        Visualize results from a prior run:
        <label for="results-file-input" class="file-input-label">
          Choose Results File
        </label>
        <input
          id="results-file-input"
          type="file"
          @change=${this.setResultsFile}
        />
      </div>
      <common-settings
        .settings=${this.settingsDefinition}
        .values=${this.settingsValues}
        @settings-changed=${this.handleSettingsChange}
      ></common-settings>

      ${
        when(
            this.results,
            () => html`<run-result-list
            .resultsRecord=${this.results}
          ></run-result-list>`,
            )}
    `;
  }

  private async setResultsFile(event: InputEvent) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      const file = input.files[0];
      const text = await file.text();
      const parsed = JSON.parse(text) as TestState;
      this.settingsValues = parsed['settings'] ?? this.settingsValues;
      this.results = parsed['results'];
    }
  }

  private downloadBatchedResults() {
    if (!this.testState?.results) {
      throw new Error('Tests not yet run');
    }

    const summary = Object.fromEntries(
        Object.entries(this.testState.results)
            .map(
                ([name, result]) =>
                    [name,
                     determineStatus(result),
    ]),
    );

    const results = {
      summary,
      results: this.testState.results,
      settings: this.testState.settings,
    };

    const serialized = JSON.stringify(results, null, 2);
    const encoded = new TextEncoder().encode(serialized);
    download(encoded.buffer, 'results.json');
  }

  private handleTabChange(event: CustomEvent) {
    const tabsElement = event.target as MdTabs;
    this.activeTabIndex = tabsElement.activeTabIndex;
  }

  override render() {
    return html`
      <div class="header">
        <h1>LiteRT Web Model Tester</h1>
        <p>
          This tool tests a LiteRT model running in WebGPU or WebNN against a set of fake
          inputs to check if there are any errors when loading or
          running the model.
        </p>
      </div>
      <md-tabs
        .activeTabIndex=${this.activeTabIndex}
        @change=${this.handleTabChange}
      >
        <md-primary-tab>Single Model</md-primary-tab>
        <md-primary-tab>Directory of Models</md-primary-tab>
      </md-tabs>

      ${choose(this.activeTabIndex, [
      [0, () => this.renderSingleModelTester()],
      [1, () => this.renderBatchedModelTester()],
    ])}
      <console-mirror></console-mirror>
    `;
  }
}

function deadline(ms: number, error?: Error): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(error), ms);
  });
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

function removeTensorData(runResult: RunResult): RunResult {
  const newResults: RunResult['results'] = {};

  for (const key of Object.keys(runResult.results)) {
    const maybeResult = runResult.results[key];
    if (!maybeResult) {
      newResults[key] = maybeResult;
      continue;
    }

    const newMaybeResult = {...maybeResult};
    if (maybeResult.value) {
      const newValue = {...maybeResult.value};
      const newTensors = {...maybeResult.value.tensors};

      if (maybeResult.value.tensors.record) {
        const newRecord: typeof maybeResult.value.tensors.record = {};
        for (const tensorKey of Object.keys(maybeResult.value.tensors.record)) {
          const tensor = maybeResult.value.tensors.record[tensorKey];
          // Copy tensor properties, but exclude the 'data' field.
          const {data, ...rest} = tensor;
          newRecord[tensorKey] = {...rest};
        }
        newTensors.record = newRecord;
      }

      if (maybeResult.value.tensors.array) {
        const newArray: typeof maybeResult.value.tensors.array =
            maybeResult.value.tensors.array.map(tensor => {
              // Copy tensor properties, but exclude the 'data' field.
              const {data, ...rest} = tensor;
              return {...rest};
            });
        newTensors.array = newArray;
      }
      newValue.tensors = newTensors;
      newMaybeResult.value = newValue;
    }
    newResults[key] = newMaybeResult;
  }

  return {...runResult, results: newResults};
}
