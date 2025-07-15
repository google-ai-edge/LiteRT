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
 * Basic styles for page-level containers and component hosts.
 */
export const hostStyles = css`
  :host {
    display: block;
    font-family: sans-serif;
    box-sizing: border-box;
  }
`;

/**
 * Styles for card-like container elements.
 */
export const cardStyles = css`
  .card {
    border: 1px solid #ccc;
    padding: 16px;
    border-radius: 8px;
    background-color: #f9f9f9;
  }
`;

/**
 * Consistent styles for typography like headings, paragraphs, and code blocks.
 */
export const typographyStyles = css`
  h1,
  h2,
  h3,
  h4 {
    margin-top: 0;
    margin-bottom: 12px;
    color: #333;
  }

  p {
    color: #555;
    line-height: 1.6;
    margin: 0;
  }

  code,
  pre {
    background-color: #eee;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-word;
  }

  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .error {
    color: #d9534f;
    background-color: #f2dede;
    border: 1px solid #ebccd1;
    padding: 8px;
    border-radius: 4px;
    margin-top: 5px;
    word-break: break-word;
  }
`;

/**
 * Shared styles for form elements like buttons and inputs.
 */
export const formStyles = css`
  button,
  .file-input-label {
    background-color: #eee;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 0.9em;
    cursor: pointer;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
    margin-right: 8px;
  }

  button:hover,
  .file-input-label:hover {
    background-color: #ddd;
    border-color: #bbb;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  button.primary {
    background-color: #343a40;
    color: white;
    border-color: #343a40;
  }

  button.primary:hover {
    background-color: #495057;
  }

  input[type='file'] {
    display: none;
  }

  input[type='text'],
  input[type='number'] {
    width: 100%;
    padding: 6px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
    font-size: inherit;
  }

  input[type='checkbox'] {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }
`;
