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

/**
 * Recursively finds all files within a directory handle that end with a
 * specific extension.
 *
 * Assumes read permission has already been granted for the initial directory
 * handle and any subdirectories encountered. Handles errors during iteration
 * gracefully.
 *
 * @param dirHandle The FileSystemDirectoryHandle for the root directory to
 *     search.
 * @param extension The desired file extension (e.g., ".txt", "tflite", ".JPG").
 *     Case-insensitive.
 * @return A Promise that resolves with an array of relative paths (strings) to
 *     the matching files.
 * Paths use forward slashes ('/') as separators.
 */
export async function findFilesByExtension(
    dirHandle: FileSystemDirectoryHandle,
    extension: string|string[]): Promise<string[]> {
  const matchingPaths: string[] = [];

  const extensionList = (extension instanceof Array ? extension : [
                          extension
                        ]).map((extension) => extension.toLowerCase());

  function hasDesiredExtension(p: string) {
    for (const extension of extensionList) {
      if (p.endsWith(extension)) {
        return true;
      }
    }
    return false;
  }

  async function traverse(
      currentHandle: FileSystemDirectoryHandle,
      currentPath: string  // Path relative to the initial dirHandle
      ): Promise<void> {
    try {
      // Iterate through entries (files and subdirectories) in the current
      // directory handle.values() returns an async iterator
      for await (const entry of currentHandle.values()) {
        // Construct the full relative path for this entry
        // If currentPath is empty (root), path is just entry.name
        // Otherwise, it's prefix/entry.name
        const entryPath =
            currentPath ? `${currentPath}/${entry.name}` : entry.name;

        if (entry.kind === 'file') {
          // If it's a file, check its extension
          if (hasDesiredExtension(entry.name.toLowerCase())) {
            // If it matches, add its relative path to the results
            matchingPaths.push(entryPath);
          }
        } else if (entry.kind === 'directory') {
          // If it's a directory, recurse into it
          // Pass the directory handle (`entry`) and the updated path prefix
          // (`entryPath`)
          await traverse(entry as FileSystemDirectoryHandle, entryPath);
        }
        // We ignore other entry kinds if any exist
      }
    } catch (error) {
      // Log errors encountered during iteration (e.g., permission issues deeper
      // in the tree) You might want to customize error handling (e.g., collect
      // errors, stop traversal)
      console.error(
          `Error accessing contents of directory "${
              currentPath || currentHandle.name}":`,
          error);
      // Continue traversal if possible, skipping the problematic
      // directory/entry
    }
  }

  await traverse(dirHandle, '');
  return matchingPaths;
}

/**
 * Returns a FileHandle for a file within a directory handle.
 *
 * @param rootDirHandle The FileSystemDirectoryHandle to search within.
 * @param path The path to the file, relative to the root directory.
 * @return A Promise that resolves with a FileHandle for the file.
 * Paths use forward slashes ('/') as separators.
 */
export async function getFileHandle(
    rootDirHandle: FileSystemDirectoryHandle, path: string) {
  const parts = path.split('/');
  const fileName = parts.pop();
  if (!fileName) {
    throw new Error('No file name found in path ' + path);
  }

  let handle = rootDirHandle;
  for (const part of parts) {
    handle = await handle.getDirectoryHandle(part);
  }

  return handle.getFileHandle(fileName);
}