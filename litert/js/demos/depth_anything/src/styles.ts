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

import {css} from 'lit';

/**
 * Styles for the image upscaler component.
 */
export const componentStyles = css`
  :host {
    display: block;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 1rem;
    color: #333;
  }
  .container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
  }
  h1 {
    color: #1a73e8;
    margin-bottom: 0;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    padding: 1rem;
    background: #f1f3f4;
    border-radius: 8px;
    width: 100%;
    box-sizing: border-box;
    align-items: end;
    justify-content: center;
  }
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  label {
    font-size: 0.8rem;
    color: #5f6368;
  }
  select, button {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    border: 1px solid #dadce0;
    font-size: 1rem;
    cursor: pointer;
  }
  button {
    background: #1a73e8;
    color: white;
    border: none;
    font-weight: 500;
  }
  button:disabled {
    background: #e0e0e0;
    cursor: not-allowed;
  }
  input[type="range"] {
    width: 150px;
  }
  .drop-zone {
    width: 100%;
    min-height: 300px;
    border: 2px dashed #dadce0;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.2s;
    overflow: hidden;
  }
  .drop-zone:hover {
    background-color: #f8f9fa;
    border-color: #1a73e8;
  }
  .drop-zone p {
    color: #5f6368;
    text-align: center;
  }
  .drop-zone img,
  .drop-zone canvas {
    max-width: 100%;
    max-height: 70vh;
    object-fit: contain;
  }
  .footer {
    width: 100%;
    text-align: center;
  }
  .status {
    min-height: 1.2em;
    color: #5f6368;
  }
  progress {
    width: 100%;
  }
  .license-info {
    font-size: 0.75rem;
    color: #5f6368;
    margin-top: 0.5rem;
    text-align: center;
  }
  .license-info a {
    color: #1a73e8;
    text-decoration: none;
  }
  .license-info a:hover {
    text-decoration: underline;
  }
  .view-original {
    margin-top: 1rem;
    background-color: #5f6368;
  }
`;