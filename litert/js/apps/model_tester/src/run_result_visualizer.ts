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

import './console_renderer';

import * as d3 from 'd3';
import {css, html, LitElement, TemplateResult} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';
import {map} from 'lit/directives/map.js';
import {when} from 'lit/directives/when.js';

import {cardStyles, hostStyles, typographyStyles} from './common_styles';
import type {BenchmarkSample, Maybe, ModelResult, RunResult, SerializableTensor} from './model_runner';
import {LITERT_WASM_CPU, LITERT_WASM_GPU} from './model_runner';

interface BenchmarkChartData {
  name: string;
  mean: number;
  stdDev: number;
  samples?: BenchmarkSample[];
  color: string;
}

interface TooltipData {
  visible: boolean;
  content: TemplateResult;
  left: number;
  top: number;
}

/* tslint:disable:no-new-decorators */

/**
 * Visualizes the results of running a model.
 */
@customElement('run-result-visualizer')
export class RunResultVisualizer extends LitElement {
  @property({type: Object}) runResult?: RunResult;

  @state() private isConsoleOpen = false;
  @state() private chartData: BenchmarkChartData[] = [];
  @state()
  private tooltipData: TooltipData = {
    visible: false,
    content: html``,
    left: 0,
    top: 0,
  };

  @query('#benchmark-chart-container') private chartContainer!: HTMLDivElement;
  @query('#benchmark-tooltip') private tooltipElement!: HTMLDivElement;

  static override styles = [
    hostStyles,
    cardStyles,
    typographyStyles,
    css`
      :host {
        /* Apply card styles to the host itself */
        border: 1px solid #ccc;
        padding: 16px;
        border-radius: 8px;
        background-color: #f9f9f9;
        margin: 1em 0;
      }
      h3 {
        border-bottom: 1px solid #eee;
        padding-bottom: 4px;
      }
      .section {
        margin-bottom: 16px;
        padding-left: 10px;
        border-left: 3px solid var(--section-border-color, #eee);
      }
      .detail {
        margin-bottom: 6px;
        color: #555;
        font-size: 0.95em;
      }
      .detail strong {
        color: #333;
      }
      ul {
        /* Overriding shared style to add disc style back */
        list-style-type: disc;
        margin-left: 20px;
      }
      li {
        margin-bottom: 4px;
      }

      /* Console Dropdown Styles */
      details {
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-top: 10px;
        background-color: #fff;
      }
      summary {
        padding: 8px 12px;
        cursor: pointer;
        font-weight: bold;
        background-color: #f0f0f0;
      }
      details[open] summary {
        border-bottom: 1px solid #ddd;
      }
      /* Benchmark Chart Styles */
      .benchmark-chart-section {
        margin-top: 20px;
      }
      #benchmark-chart-container {
        width: 100%;
        min-height: 250px;
        background-color: #fff;
        padding: 10px;
        box-sizing: border-box;
        border: 1px solid #ddd;
        border-radius: 4px;
        position: relative; /* For tooltip positioning */
      }
      .benchmark-chart-section .axis path,
      .benchmark-chart-section .axis line {
        fill: none;
        stroke: #888;
        shape-rendering: crispEdges;
      }
      .benchmark-chart-section .axis text {
        font-size: 0.8em;
        fill: #555;
      }
      .benchmark-chart-section .bar {
        shape-rendering: crispEdges;
      }
      .benchmark-chart-section .bar:hover {
        opacity: 0.8;
      }
      .benchmark-chart-section .bar-label {
        font-size: 0.85em;
        font-weight: bold;
        fill: #333;
      }
      .benchmark-chart-section .error-bar {
        stroke: #333;
        stroke-width: 1.5px;
      }
      .benchmark-chart-section .error-cap {
        stroke: #333;
        stroke-width: 1.5px;
      }
      .chart-title {
        font-size: 1.1em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        color: #333;
      }
      #benchmark-tooltip {
        position: absolute;
        text-align: left;
        padding: 6px 8px;
        font-size: 0.85em;
        background: rgba(240, 240, 240, 0.95);
        border: 1px solid #ccc;
        border-radius: 4px;
        pointer-events: none;
        opacity: 0;
        white-space: nowrap;
        z-index: 10; /* Ensure tooltip is on top */
        transition: opacity 0.2s ease-in-out;
        line-height: 1.4;
      }
      #benchmark-tooltip.visible {
        opacity: 1;
      }
      .tooltip-content strong {
        color: #333;
      }
    `,
  ];

  /**
   * Colors for the accelerators. Other accelerators will be assigned random but
   * consistent colors.
   */
  static AcceleratorColors: Record<string, string> = {
    [LITERT_WASM_CPU]: '#64b5f6',
    [LITERT_WASM_GPU]: '#81c784',
  };

  protected override willUpdate(
      changedProperties: Map<string|number|symbol, unknown>) {
    // Keep chartData up to date with the incoming RunResult.
    if (changedProperties.has('runResult')) {
      if (this.runResult) {
        this.chartData = prepareChartData(this.runResult);
      } else {
        this.chartData = [];
      }
    }
  }

  protected override updated(
      changedProperties: Map<string|number|symbol, unknown>) {
    // The d3 chart must be rendered after the update so that the div it uses
    // for rendering is in the DOM.
    if (changedProperties.has('chartData')) {
      if (this.chartData.length > 0 && this.chartContainer) {
        this.renderBenchmarkComparisonChart(
            this.chartContainer, this.chartData);
      } else if (this.chartContainer) {
        d3.select(this.chartContainer).select('svg').remove();
      }
    }
  }

  private renderBenchmarkComparisonChart(
      container: HTMLElement, data: BenchmarkChartData[]): void {
    d3.select(container).select('svg').remove();
    d3.select(container).select('.tooltip').remove();  // Clear previous tooltip
    const margin = {top: 40, right: 30, bottom: 50, left: 60};
    const availableWidth = container.clientWidth;
    const width = Math.max(
        200,
        availableWidth - margin.left - margin.right);  // Ensure minimum width
    const height = 250 - margin.top - margin.bottom;

    const svg =
        d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 0 - margin.top / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('class', 'chart-title')
        .text('Backend Mean Run Time (ms)');

    const x = d3.scaleBand()
                  .range([0, width])
                  .domain(data.map(d => d.name))
                  .padding(0.3);
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    const yMax = d3.max(data, d => d.mean + d.stdDev);
    const y =
        d3.scaleLinear().domain([0, yMax ? yMax * 1.1 : 10]).range([height, 0]);
    svg.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d => `${d} ms`));

    // Draw bars
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => x(d.name)!)
        .attr('y', d => y(d.mean))
        .attr('width', x.bandwidth())
        .attr('height', d => height - y(d.mean))
        .attr('fill', d => d.color)
        .on('mouseenter', ((d: BenchmarkChartData) => {
              const tooltipContent = html`
                <div class="tooltip-content">
                  <strong>${d.name}</strong><br/>
                  Mean: ${d.mean.toFixed(2)} ms<br/>
                  StdDev: ${d.stdDev.toFixed(2)} ms<br/>
                  Samples: ${d.samples?.length || 'N/A'}
                </div>
              `;
              this.tooltipData = {
                visible: true,
                content: tooltipContent,
                left: d3.event.offsetX + 10,
                top: d3.event.offsetY - 20,
              };
            }))
        .on('mouseleave', () => {
          this.tooltipData = {...this.tooltipData, visible: false};
        });
    svg.selectAll('.bar-label')
        .data(data)
        .enter()
        .append('text')
        .attr('class', 'bar-label')
        .attr('x', d => x(d.name)! + x.bandwidth() / 2)
        .attr(
            'y',
            d => {
              // Position label inside the bar if it's too tall, otherwise above
              const barTopPosition = y(d.mean);
              return barTopPosition < 20 ? barTopPosition + 15 :
                                           barTopPosition - 5;
            })
        .attr('fill', d => '#333')  // Set the color of the text
        .attr('text-anchor', 'middle')
        .text(d => `${d.mean.toFixed(1)} ms`);

    // Draw error bars
    data.forEach(d => {
      const barX = x(d.name)! + x.bandwidth() / 2;
      const yTop = y(d.mean + d.stdDev);
      const yBottom = y(d.mean - d.stdDev);
      if (d.stdDev > 0) {
        // Only draw if stdDev is meaningful
        svg.append('line')
            .attr('class', 'error-bar')
            .attr('x1', barX)
            .attr('x2', barX)
            .attr('y1', yTop)
            .attr('y2', yBottom);
        svg.append('line')
            .attr('class', 'error-cap')
            .attr('x1', barX - 5)
            .attr('x2', barX + 5)
            .attr('y1', yTop)
            .attr('y2', yTop);
        svg.append('line')
            .attr('class', 'error-cap')
            .attr('x1', barX - 5)
            .attr('x2', barX + 5)
            .attr('y1', yBottom)
            .attr('y2', yBottom);
      }
    });
  }

  private formatTensorInfo(tensor: SerializableTensor): string {
    if (tensor && Array.isArray(tensor.shape) &&
        typeof tensor.dtype === 'string') {
      const shapeString =
          tensor.shape.length > 0 ? tensor.shape.join('x') : 'scalar';
      return `${shapeString}x${tensor.dtype}`;
    }
    return 'Invalid/Missing Tensor Data';
  }

  private renderFailedResults(results: Record<string, Maybe<ModelResult>>) {
    const failedResults = [];
    for (const [name, maybeResult] of Object.entries(results)) {
      if (maybeResult.value) {
        continue;
      }
      failedResults.push(html`
        <div class="detail error" style="margin-top: 5px;">
          ${name} Benchmark Error:
          <pre>${maybeResult.error}</pre>
        </div>`);
    }
    return failedResults;
  }

  override render() {
    if (!this.runResult) {
      return html`<p>No RunResult data provided.</p>`;
    }

    const failedResults = this.renderFailedResults(this.runResult.results);
    const {results, consoleMessages} = this.runResult;

    // The source of tensor data for rendering output shapes and names.
    // This is CPU as long as CPU succeeded. Otherwise, it's the first backend
    // that succeeded.
    let primaryTensorSource: ModelResult|undefined =
        results[LITERT_WASM_CPU]?.value;
    if (!primaryTensorSource) {
      for (const maybeResult of Object.values(results)) {
        if (maybeResult.value) {
          primaryTensorSource = maybeResult.value;
          break;
        }
      }
    }
    const compareAgainstResult = results[LITERT_WASM_GPU]?.value;
    const mse = compareAgainstResult?.meanSquaredError;

    return html`
      <div
        class="section benchmark-chart-section"
        style="--section-border-color: #7e57c2;"
      >
        <h3>Benchmark Performance</h3>
        ${
        when(
            this.chartData.length > 0,
            () => html`
          <div id="benchmark-chart-container">
            <div
              id="benchmark-tooltip"
              class="${this.tooltipData.visible ? 'visible' : ''}"
              style="left: ${this.tooltipData.left}px; top: ${
                this.tooltipData.top}px;"
            >
              ${this.tooltipData.content}
            </div>
          </div>
        `,
            () => html`
          <div class="detail">No benchmark data available.</div>
        `)}
        ${failedResults}
      </div>

      <div class="section" style="--section-border-color: #ffd54f;">
        <h3>Outputs</h3>
        ${
        primaryTensorSource ?
            html`
              <ul>
                ${
                map(Object.entries(primaryTensorSource.tensors.record),
                    ([key, tensor]) => html`
                  <li>
                    <code>${key}</code>: ${this.formatTensorInfo(tensor)} 
                    ${
                        mse && mse[key] !== undefined ?
                            html` - MSE: ${mse[key].toExponential(2)}` :
                            ''}
                  </li>
                `)}
              </ul>
            ` :
            html`
              <div class="detail">
                ${
                failedResults.length > 0 ?
                    'Output data could not be determined due to errors.' :
                    primaryTensorSource ?
                    'No output tensor data available.' :
                    'No output tensor data available and no successful runs.'}
              </div>
            `}
      </div>

      ${
        when(
            consoleMessages && consoleMessages.length > 0,
            () => html`
        <div class="section" style="--section-border-color: #b0bec5;">
          <h3>Console Output</h3>
          <details
            @toggle=${
                (e: Event) => (
                    this.isConsoleOpen = (e.target as HTMLDetailsElement).open)}
          >
            <summary>
              Show/Hide Console (${consoleMessages!.length} messages)
            </summary>
            <console-renderer
              .messages=${consoleMessages!}
            ></console-renderer>
          </details>
        </div>
      `)}
    `;
  }
}

function calculateBenchmarkStats(samples: BenchmarkSample[]):
    {mean: number, stdDev: number} {
  const mean =
      samples.reduce((sum, {latency}) => sum + latency, 0) / samples.length;
  const stdDev = Math.sqrt(
      samples.reduce((sum, {latency}) => sum + latency * latency, 0) /
          samples.length -
      mean * mean);
  return {mean, stdDev};
}

function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  let color = '#';
  for (let i = 0; i < 3; i++) {
    const value = (hash >> (i * 8)) & 0xff;
    const hex = value.toString(16).padStart(2, '0');
    color += hex;
  }
  return color;
}

function prepareChartData(runResult: RunResult): BenchmarkChartData[] {
  const chartData: BenchmarkChartData[] = [];

  for (const [name, maybeResult] of Object.entries(runResult.results)) {
    const benchmark = maybeResult?.value?.benchmark;
    if (!benchmark) {
      continue;
    }
    const {mean, stdDev} = calculateBenchmarkStats(benchmark.samples);
    chartData.push({
      name,
      mean,
      stdDev,
      samples: benchmark.samples,
      color: RunResultVisualizer.AcceleratorColors[name] ?? stringToColor(name),
    });
  }

  return chartData;
}
