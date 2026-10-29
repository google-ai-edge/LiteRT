# Copyright 2026 The LiteRT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inventory Vulkan usage against pinned registry objects, retaining review gaps.

Never infer conformance from an API call's presence or a passing numerical test.
The reviewed manifest classifies individual VUIDs, and stale/incomplete reviews
fail --check. Generated inventory and reports belong outside the source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]
SOURCES = [BACKEND / 'dispatch' / name for name in
           ['vulkan_invocation.cc', 'device_capabilities.cc', 'tensor_capabilities.cc',
            'vulkan_validation.cc', 'vulkan_validation.h', 'device_memory_allocator.h',
            'external_buffer.cc', 'external_buffer.h']]
SOURCES.append(BACKEND / 'tools/devtool/tool.cc')

def inventory(registry):
    raw = (registry / 'validusage.json').read_bytes()
    database = json.loads(raw)
    xml = ET.parse(registry / 'vk.xml').getroot()
    text = '\n'.join(p.read_text() for p in SOURCES)
    names = set(re.findall(r'\bVk\w+', text))
    names.update(re.findall(r'\bPFN_(vk\w+)', text))
    names.update(re.findall(r'\b(?:INSTANCE_FN|DEVICE_FN)\((\w+)\)', text))
    names = {('vk' + n if not n.startswith(('Vk', 'vk')) else n) for n in names}
    # Include command parameter types and nested members of structures used by
    # the source, not only structure names spelled directly at the call site.
    definitions = {t.get('name', t.findtext('name')): t for t in xml.findall('./types/type')}
    commands = {c.findtext('./proto/name'): c for c in xml.findall('./commands/command') if c.find('proto') is not None}
    pending = list(names)
    while pending:
        name = pending.pop()
        element = commands.get(name)
        if element is None:
            element = definitions.get(name)
        if element is None:
            continue
        dependencies = [x.text for x in element.findall('.//type') if x.text]
        alias = element.get('alias')
        if alias:
            dependencies.append(alias)
        for dependency in dependencies:
            if dependency.startswith('Vk') and dependency not in names:
                names.add(dependency)
                pending.append(dependency)
    # Runtime SPIR-V rules need explicit applicability review too; core shader
    # rules must not be silently omitted just because GraphARM is used.
    names.add('RuntimeSpirv')
    rows = {}
    for name in sorted(names):
        for group, entries in database['validation'].get(name, {}).items():
            for entry in entries:
                rows[entry['vuid']] = dict(owner=name, group=group, text=entry['text'], page=entry.get('page'))
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(), version=database['version info'],
                source_files={str(p.relative_to(BACKEND)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
                owners=sorted(names), requirements=rows)

def review(result, manifest):
    errors = []
    if manifest.get('source_sha256') != result['source_sha256']:
        errors.append('Pinned validusage.json hash differs from reviewed source')
    if manifest.get('source_files') != result['source_files']:
        errors.append('Vulkan source inventory changed; review new usage/conditions')
    records = dict(manifest.get('requirements', {}))
    for group in manifest.get('groups', []):
        for vuid in group['vuids']:
            if vuid in records:
                errors.append(vuid + ': duplicate classification')
            records[vuid] = group
    for vuid, requirement in result['requirements'].items():
        record = records.get(vuid, {})
        state = record.get('status')
        if state not in ['tested', 'enforced', 'inapplicable', 'external_failure']:
            errors.append(vuid + ': unreviewed')
        elif not record.get('reason') or not record.get('evidence'):
            errors.append(vuid + ': missing applicability/enforcement evidence')
        if state == 'tested' and not record.get('tests'):
            errors.append(vuid + ': missing test target')
    for vuid in records.keys() - result['requirements'].keys():
        errors.append(vuid + ': stale review entry')
    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=HERE / 'vuid_review.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = inventory(args.registry)
    manifest = json.loads(args.review.read_text()) if args.review.exists() else {}
    errors = review(result, manifest)
    result['review_errors'] = errors
    result['complete'] = not errors
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(result['requirements'])} VUIDs, {len(errors)} review gaps; {args.output}")
    if args.check and errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
