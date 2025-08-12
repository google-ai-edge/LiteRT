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

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {map} from 'lit/directives/map.js';
import {when} from 'lit/directives/when.js';

import {hostStyles} from './common_styles';

/**
 * A message from the console.
 */
export interface ConsoleMessage {
  type: 'log'|'warn'|'error'|'info'|'debug';
  message: string;
  timestamp?: number;
}

/* tslint:disable:no-new-decorators */

/**
 * A component that renders the console messages.
 */
@customElement('console-renderer')
export class ConsoleRenderer extends LitElement {
  @property({type: Array}) messages: ConsoleMessage[] = [];

  static override styles = [
    hostStyles,
    css`
      :host {
        display: block;
        padding: 10px;
        background-color: #282c34;
        color: #abb2bf;
        font-family: monospace;
        font-size: 0.9em;
        max-height: 300px;
        overflow-y: auto;
      }
      .console-message {
        padding: 2px 0;
        border-bottom: 1px dotted #444;
        line-height: 1.4;
      }
      .console-message:last-child {
        border-bottom: none;
      }
      .timestamp {
        color: #61afef;
        margin-right: 8px;
      }
      .type-log {
        color: #98c379;
      }
      .type-warn {
        color: #e5c07b;
      }
      .type-error {
        color: #e06c75;
      }
      .type-info {
        color: #61afef;
      }
      .type-debug {
        color: #c678dd;
      }
      .message-content {
        white-space: pre-wrap;
      }
    `,
  ];

  override render() {
    if (!this.messages || this.messages.length === 0) {
      return html`<div class="console-message">
        No console messages recorded.
      </div>`;
    }

    return html`
      ${map(this.messages, (msg) => html`
          <div class="console-message">
            ${when(msg.timestamp, () => html`<span class="timestamp"
                  >[${new Date(msg.timestamp!).toISOString()}]</span
                >`)}
            <span class="type-${msg.type.toLowerCase()}"
              >[${msg.type.toUpperCase()}]</span
            >
            <span class="message-content">${msg.message}</span>
          </div>
        `)}
    `;
  }
}
