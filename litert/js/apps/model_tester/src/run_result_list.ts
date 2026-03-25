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

import './run_result_visualizer';

import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {map} from 'lit/directives/map.js';
import {when} from 'lit/directives/when.js';

import {cardStyles, formStyles, hostStyles, typographyStyles} from './common_styles';
import {LITERT_WASM_CPU, LITERT_WASM_GPU, LITERT_WASM_WEBNN, Maybe, ModelResult, RunResult} from './model_runner';

const MSE_THRESHOLD = 1e-4;

/**
 * The status of a run result, used to color the UI and in the JSON summary.
 */
export type Status =|'all passed'|'some passed'|'none passed'|'not run';

const statusClassMap: Record<Status, string> = {
  'all passed': 'status-green',
  'some passed': 'status-orange',
  'none passed': 'status-red',
  'not run': 'status-black',
};

const legendItems:
    ReadonlyArray<{text: string; class: string; statusKey: Status;}> = [
      {
        text: 'All backends passed',
        class: statusClassMap['all passed'],
        statusKey: 'all passed',
      },
      {
        text: 'Some backends passed',
        class: statusClassMap['some passed'],
        statusKey: 'some passed',
      },
      {
        text: 'No backends passed',
        class: statusClassMap['none passed'],
        statusKey: 'none passed',
      },
      {text: 'Not Run', class: statusClassMap['not run'], statusKey: 'not run'},
    ] as const;

/**
 * Determines the status of a run result.
 *
 * @param runResult The run result to determine the status of.
 * @return The status of the run result.
 */
export function determineStatus(runResult: RunResult|undefined): Status {
  if (!runResult) {
    return 'not run';
  }

  const {results} = runResult;

  if (Object.keys(results).length === 0) {
    return 'not run';
  }

  const backends = [LITERT_WASM_CPU, LITERT_WASM_GPU, LITERT_WASM_WEBNN];
  let passCount = 0;

  for (const backend of backends) {
    const result = results[backend];
    if (result && !result.error) {
      let mseOk = true;
      if (result.value?.meanSquaredError) {
        const mse = result.value.meanSquaredError;
        mseOk = Object.keys(mse).length > 0 &&
            Object.values(mse).every((val) => val < MSE_THRESHOLD);
      }
      if (mseOk) {
        passCount++;
      }
    }
  }

  if (passCount === backends.length) {
    return 'all passed';
  } else if (passCount === 0) {
    return 'none passed';
  } else {
    return 'some passed';
  }
}

/* tslint:disable:no-new-decorators */
/**
 * A list of run results, with filtering and result visualization.
 */
@customElement('run-result-list')
export class RunResultList extends LitElement {
  @property({type: Object}) resultsRecord?: Record<string, RunResult>;

  @state() private _selectedStatuses = new Set<Status>();

  @state()
  private _expandedResults =
      new Set<string>();  // Stores modelNames of expanded items



  static override styles = [
    // Import shared styles
    hostStyles,
    cardStyles,
    typographyStyles,
    formStyles,
    // Component-specific styles
    css`
      :host {
        /* Apply card styles to the host itself */
        border: 1px solid #ccc;
        padding: 16px;
        border-radius: 8px;
        background-color: #f9f9f9;
      }
      .results-list > li {
        border-bottom: 1px solid #eee;
      }
      .results-list > li:last-child {
        border-bottom: none;
      }

      .result-item-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 5px;
        cursor: pointer;
        transition: background-color 0.2s ease;
      }
      .result-item-header:hover {
        background-color: #f0f0f0;
      }

      .status-icon {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
      }
      .model-name {
        font-weight: normal;
        word-break: break-all;
        flex-grow: 1;
      }

      /* Status colors are specific to this component */
      .status-green {
        background-color: #28a745;
      }
      .status-orange {
        background-color: #fd7e14;
      }
      .status-red {
        background-color: #dc3545;
      }
      .status-black {
        background-color: #343a40;
      }

      /* Legend styles are specific to this component */
      .legend-container {
        margin-bottom: 16px;
        padding: 10px;
        border: 1px dashed #ddd;
        border-radius: 4px;
        background-color: #fff;
      }
      .legend-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 5px 10px;
      }
      .legend-item {
        padding: 4px 0;
        font-size: 0.9em;
        display: flex;
        align-items: center;
        gap: 5px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.2s ease, opacity 0.2s ease;
        border: 1px solid transparent;
      }
      .legend-item:hover {
        background-color: #f0f0f0;
      }
      .legend-item.selected {
        background-color: #e0e8f0;
        border: 1px solid #aaccf0;
        opacity: 1 !important;
      }
      .legend-container.filter-active .legend-item:not(.selected) {
        opacity: 0.6;
      }
      .legend-item .status-text {
        flex-grow: 0;
      }
      .legend-item .percentage {
        flex-grow: 1;
        font-weight: bold;
        color: #555;
        padding-left: 5px;
      }
      button.clear-filter-button {
        margin-top: 10px;
      }

      /* visualizer is now styled by its own :host styles */
      run-result-visualizer {
        margin: 0 5px 10px 20px;
      }

      /* Styles for aligned performance time columns */
      .perf-times {
        display: flex;
        gap: 4px;
        margin-left: auto;
        flex-shrink: 0;
        font-size: 0.85em;
        text-align: left;
      }

      .perf-time {
        width: 100px; /* Fixed width for alignment */
        padding: 2px 6px;
        border-radius: 4px;
        white-space: nowrap;
        color: #333;
        box-sizing: border-box;
      }

      .cpu-time {
        background-color: #e3f2fd; /* Light blue */
      }

      .gpu-time {
        background-color: #e8f5e9; /* Light green */
      }

      .webnn-time {
        background-color: #f3e5f5; /* Light purple */
      }
    `,
  ];

  private determineStatusClass(runResult: RunResult|undefined): string {
    return statusClassMap[determineStatus(runResult)];
  }

  private _toggleLegendItem(statusKey: Status): void {
    const newSelectedStatuses = new Set(this._selectedStatuses);
    if (newSelectedStatuses.has(statusKey)) {
      newSelectedStatuses.delete(statusKey);
    } else {
      newSelectedStatuses.add(statusKey);
    }
    this._selectedStatuses = newSelectedStatuses;
  }

  private _toggleResultExpansion(modelName: string): void {
    const newExpandedResults = new Set(this._expandedResults);
    if (newExpandedResults.has(modelName)) {
      newExpandedResults.delete(modelName);
    } else {
      newExpandedResults.add(modelName);
    }
    this._expandedResults = newExpandedResults;
  }

  override render() {
    const resultsArray =
        this.resultsRecord ? Object.values(this.resultsRecord) : [];
    const totalResults = resultsArray.length;
    const hasResults = totalResults > 0;
    const entries =
        this.resultsRecord ? Object.entries(this.resultsRecord) : [];

    const statusCounts = new Map<Status, number>();
    legendItems.forEach((item) => statusCounts.set(item.statusKey, 0));

    if (hasResults) {
      resultsArray.forEach((runResult) => {
        const status = determineStatus(runResult);
        statusCounts.set(status, (statusCounts.get(status) || 0) + 1);
      });
    }

    const isFilterActive = this._selectedStatuses.size > 0;
    const filteredEntries = isFilterActive ?
        entries.filter(([_, runResult]) => {
          const status = determineStatus(runResult);
          return this._selectedStatuses.has(status);
        }) :
        entries;

    return html`
      <div
        class="legend-container ${classMap({
      'filter-active': isFilterActive
    })}"
      >
        <h4>Legend</h4>
        <ul class="legend-list">
          ${
        map(legendItems,
            (item) => {
              const count = statusCounts.get(item.statusKey) || 0;
              const percentage = totalResults > 0 ?
                  ((count / totalResults) * 100).toFixed(0) :
                  0;
              const isSelected = this._selectedStatuses.has(item.statusKey);
              const itemClasses = {
                'legend-item': true,
                selected: isSelected,
              };
              return html`
              <li
                class=${classMap(itemClasses)}
                @click=${() => this._toggleLegendItem(item.statusKey)}
                title="Click to filter by '${item.text}' status"
              >
                <span class="status-icon ${item.class}"></span>
                <span class="status-text">${item.text}</span>
                <span class="percentage">(${percentage}%)</span>
              </li>
            `;
            })}
        </ul>
        ${when(isFilterActive, () => html`
            <button
              class="clear-filter-button"
              @click=${() => {
                                 this._selectedStatuses = new Set();
                               }}
            >
              Clear Filter
            </button>
          `)}
      </div>

      ${
        when(
            hasResults,
            () => html`
          <ul class="results-list">
            ${
                filteredEntries.length > 0 ?
                    map(filteredEntries,
                        ([modelName, runResult]) => {
                          const statusClass =
                              this.determineStatusClass(runResult);
                          const isExpanded =
                              this._expandedResults.has(modelName);

                          const results = runResult?.results;
                          const webnnTime =
                              meanLatency(results?.[LITERT_WASM_WEBNN]);
                          const gpuTime =
                              meanLatency(results?.[LITERT_WASM_GPU]);
                          const cpuTime =
                              meanLatency(results?.[LITERT_WASM_CPU]);

                          const cpuTimeText = cpuTime != null ?
                              `${cpuTime.toFixed(1)}ms` :
                              '---';
                          const gpuTimeText = gpuTime != null ?
                              `${gpuTime.toFixed(1)}ms` :
                              '---';
                          const webnnTimeText = webnnTime != null ?
                              `${webnnTime.toFixed(1)}ms` :
                              '---';

                          const perfInfo = html`
                    <div class="perf-times">
                      <span class="perf-time cpu-time" title="CPU Time (mean)"
                        >CPU: ${cpuTimeText}</span
                      >
                      <span class="perf-time gpu-time" title="GPU Time (mean)"
                        >GPU: ${gpuTimeText}</span
                      >
                      <span class="perf-time webnn-time" title="WebNN Time (mean)"
                        >WebNN: ${webnnTimeText}</span
                      >
                    </div>
                  `;

                          return html`
                    <li>
                      <div
                        class="result-item-header"
                        @click=${() => this._toggleResultExpansion(modelName)}
                      >
                        <span class="status-icon ${statusClass}"></span>
                        <span class="model-name">${modelName}</span>
                        ${perfInfo}
                      </div>
                      ${when(isExpanded, () => html`
                          <run-result-visualizer
                            id="details-${modelName}"
                            .runResult=${runResult}
                          ></run-result-visualizer>
                        `)}
                    </li>
                  `;
                        }) :
                    html`<li>No results match the selected filter.</li>`}
          </ul>
        `,
            () => html` <p>No results data available.</p> `)}
    `;
  }
}

function meanLatency(modelResult: Maybe<ModelResult>): number|undefined {
  const samples = modelResult?.value?.benchmark?.samples?.map(s => s.latency);
  if (!samples || samples.length === 0) {
    return undefined;
  }
  const sum = samples.reduce((a, b) => a + b, 0);
  return sum / samples.length;
}