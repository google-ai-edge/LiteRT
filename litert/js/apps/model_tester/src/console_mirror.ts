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

import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

import {hostStyles} from './common_styles';
import type {ConsoleMessage} from './console_renderer';

/* tslint:disable:no-new-decorators */

/**
 * A component that mirrors the console output to a custom element.
 * It intercepts calls to console.log, console.error, and console.warn.
 */
@customElement('console-mirror')
export class ConsoleMirror extends LitElement {
  @state() private messages: ConsoleMessage[] = [];

  @property({type: Boolean}) autoScroll = true;

  static override styles = [
    hostStyles,
    css`
      :host {
        display: block;
      }
      console-renderer {
        border: 1px solid #ccc;
        font-size: 0.9em;
        max-height: 600px;
      }
    `,
  ];

  private originalLog: (...args: unknown[]) => void;
  private originalError: (...args: unknown[]) => void;
  private originalWarn: (...args: unknown[]) => void;

  constructor() {
    super();
    this.originalLog = console.log;
    this.originalError = console.error;
    this.originalWarn = console.warn;
  }

  getMessages() {
    return this.messages;
  }

  override connectedCallback() {
    super.connectedCallback();
    console.log = (...args: unknown[]) => {
      this.messages = [
        ...this.messages,
        {
          type: 'log',
          message: args.map((arg) => this.formatArgument(arg)).join(' '),
          timestamp: new Date().getTime(),
        },
      ];
      this.requestUpdate();
      this.originalLog(...args);
    };
    console.error = (...args: unknown[]) => {
      this.messages = [
        ...this.messages,
        {
          type: 'error',
          message: args.map((arg) => this.formatArgument(arg)).join(' '),
          timestamp: new Date().getTime(),
        },
      ];
      this.requestUpdate();
      this.originalError(...args);
    };
    console.warn = (...args: unknown[]) => {
      this.messages = [
        ...this.messages,
        {
          type: 'warn',
          message: args.map((arg) => this.formatArgument(arg)).join(' '),
          timestamp: new Date().getTime(),
        },
      ];
      this.requestUpdate();
      this.originalWarn(...args);
    };
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    console.log = this.originalLog;
    console.error = this.originalError;
    console.warn = this.originalWarn;
  }

  private formatArgument(arg: unknown): string {
    return String(arg);
  }

  override render() {
    return html` <console-renderer .messages=${
        this.messages}></console-renderer> `;
  }
}
