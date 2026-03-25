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

import {SettingValues} from './common_settings';
import {findFilesByExtension} from './file_utils';
import {RunResult} from './model_runner';

const DB_NAME = 'testData';
const STORE_NAME = 'testState';

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, 1);

    request.onerror = () =>
        reject(new Error(`Error opening database: ${request.error?.message}`));
    request.onupgradeneeded = event => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, {keyPath: 'id'});
      }
    };
    request.onsuccess = event => {
      const db = (event.target as IDBOpenDBRequest)?.result;
      if (db) {
        resolve(db);
      } else {
        reject(new Error('Database was not defined on successful open'));
      }
    };
  });
}

/**
 * The current state of a test run over a set of models.
 */
export interface TestState {
  settings?: SettingValues;
  pathsToTest: string[];
  filesystemHandle: FileSystemDirectoryHandle;
  pathIndex: number;
  results: Record<string, RunResult|undefined>;
}

/**
 * Saves the current state of a test run to the database.
 *
 * @param testState The current state of a test run.
 * @return A Promise that resolves when the state is saved.
 */
export async function saveTestState(testState: TestState) {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');

    const store = transaction.objectStore(STORE_NAME);

    const data = {
      id: 'testState',
      testState,
    };

    const request = store.put(data);

    request.onerror = reject;
    transaction.onerror = reject;
    transaction.oncomplete = resolve;
  });
}

/**
 * Loads the current state of a test run from the database.
 *
 * @return A Promise that resolves with the current state of a test run, or
 *     undefined if no state is found.
 */
export async function loadTestState(): Promise<TestState|undefined> {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    const request = store.get('testState');
    request.onsuccess = () => {
      const data = request.result;
      if (data?.testState) {
        resolve(data.testState as TestState);
      } else {
        resolve(undefined);
      }
    };

    request.onerror = reject;
    transaction.onerror = reject;
  });
}

/**
 * Selects a directory from the file picker and loads the test state from the
 * database.
 *
 * @return A Promise that resolves with the current state of a test run.
 */
export async function selectDirectory() {
  const windowWithFilesystemApi = window as unknown as {
    showDirectoryPicker: () => Promise<FileSystemDirectoryHandle>;
  };
  const directoryHandle = await windowWithFilesystemApi.showDirectoryPicker();

  const models = await findFilesByExtension(directoryHandle, [
    '.tflite',
  ]);
  const testState: TestState = {
    pathsToTest: models,
    filesystemHandle: directoryHandle,
    pathIndex: 0,
    results: Object.fromEntries(
        models.sort((a, b) => a.localeCompare(b)).map(p => [p, undefined])),
  };

  await saveTestState(testState);
  return testState;
}

/**
 * Clears the test results from the database and returns the test state, ready
 * to re-run the tests.
 *
 * @return A Promise that resolves with the current state of a test run, or
 *     undefined if no state is found.
 */
export async function clearTestResults() {
  const testState = await loadTestState();
  if (!testState) {
    return;
  }

  testState.pathIndex = 0;
  testState.results =
      Object.fromEntries(testState.pathsToTest.map(p => [p, undefined]));
  await saveTestState(testState);
  return testState;
}