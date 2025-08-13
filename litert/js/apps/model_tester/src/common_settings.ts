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
import {live} from 'lit/directives/live.js';

import {formStyles, hostStyles} from './common_styles';

/* tslint:disable:no-new-decorators */

type DataType = 'string'|'number'|'boolean';

/**
 * A definition for a setting that can be displayed and edited.
 */
export interface SettingDef {
  key: string;  // The programmatic key
  label: string;
  dataType: DataType;
}
/**
 * A list of settings.
 */
export type Settings = SettingDef[];

/**
 * A value for a setting.
 */
export interface SettingValue {
  key: string;
  value: string|number|boolean|undefined;
}
/**
 * A list of values for settings.
 */
export type SettingValues = SettingValue[];

/**
 * A component to display and edit a list of settings.
 *
 * @property {Settings} settings - The definition of the settings to display.
 * @property {SettingValues} values - The current values for the settings.
 * @fires settings-changed - Dispatched when a setting's value is changed by the
 * user. Detail: { key: string, value: any }
 * @csspart setting-row - The container for a single setting's label and
 * control.
 * @csspart setting-label - The label element for a setting.
 * @csspart setting-control - The container for the input/visualizer element.
 */
@customElement('common-settings')
export class CommonSettings extends LitElement {
  static override styles = [
    hostStyles,
    formStyles,
    css`
      /* Component-specific layout styles */
      .setting-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #eee;
        gap: 10px 15px;
        part: setting-row;
      }
      .setting-row:last-child {
        border-bottom: none;
      }
      label {
        flex-basis: 120px;
        flex-grow: 1;
        font-weight: 500;
        padding-right: 5px;
        cursor: default;
        part: setting-label;
      }
      .control {
        flex-basis: 200px;
        flex-grow: 2;
        display: flex;
        align-items: center;
        part: setting-control;
      }oc
      input[type='checkbox'] {
        margin-right: auto;
      }
      .visualizer-container {
        width: 100%;
      }
      .visualizer-container > * {
        width: 100%;
        box-sizing: border-box;
      }
    `,
  ];

  @property({type: Array}) settings: Settings = [];
  @property({type: Array}) values: SettingValues = [];

  /** Retrieves the current value for a given setting key */
  private getValue(key: string): string|number|boolean|undefined {
    return this.values.find(v => v.key === key)?.value;
  }

  /** Dispatches the settings-changed event */
  private dispatchChange(key: string, value: string|number|boolean|undefined):
      void {
    const event = new CustomEvent('settings-changed', {
      detail: {key, value},
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }

  private handleInputChange(e: Event, key: string, type: 'string'|'number'):
      void {
    const input = e.target as HTMLInputElement;
    let value: string|number|undefined = input.value;

    if (type === 'number') {
      if (input.value === '') {
        value = undefined;  // Treat empty number input as undefined
      } else {
        const numValue = Number(input.value);
        // Dispatch undefined if parsing fails (results in NaN)
        value = isNaN(numValue) ? undefined : numValue;
      }
    }
    this.dispatchChange(key, value);
  }

  private handleCheckboxChange(e: Event, key: string): void {
    const input = e.target as HTMLInputElement;
    this.dispatchChange(key, input.checked);
  }

  override render() {
    return html`
      ${this.settings.map(setting => {
      const currentValue = this.getValue(setting.key);
      let visualizerContent;

      switch (setting.dataType) {
        case 'string':
          visualizerContent = html` <input
                id=${setting.key}
                type="text"
                .value=${live((currentValue as string) ?? '')}
                @input=${
              (e: Event) => this.handleInputChange(e, setting.key, 'string')}
              />`;
          break;
        case 'number':
          visualizerContent = html` <input
                id=${setting.key}
                type="number"
                .value=${live(currentValue != null ? String(currentValue) : '')}
                @input=${
              (e: Event) => this.handleInputChange(e, setting.key, 'number')}
              />`;
          break;
        case 'boolean':
          visualizerContent = html` <input
                id=${setting.key}
                type="checkbox"
                ?checked=${(currentValue as boolean) ?? false}
                @change=${
              (e: Event) => this.handleCheckboxChange(e, setting.key)}
              />`;
          break;
        default:
          // Render nothing or an error for unsupported types
          visualizerContent = '';
      }

      return html`
          <div class="setting-row" part="setting-row">
            <label for=${setting.key} part="setting-label"
              >${setting.label}</label
            >
            <div class="control" part="setting-control">
              ${visualizerContent}
            </div>
          </div>
        `;
    })}
  `;
  }
}
