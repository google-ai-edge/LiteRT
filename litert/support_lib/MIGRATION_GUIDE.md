# Migrating from TensorFlow Lite Support to LiteRT Support

The TensorFlow Lite Support Library and TensorFlow Lite Metadata Library now
ship as **LiteRT Support** and **LiteRT Metadata**, under the
`com.google.ai.edge.litert` Maven group.

The public API hasn't changed. Class names, methods, and behavior are the same.
For most apps, migrating takes two steps:

1.  Swap the Gradle coordinates.
2.  Rename the `org.tensorflow.lite.support` package to
    `com.google.ai.edge.litert.support`.

## 1. Artifact mapping

| TensorFlow Lite (old)                        | LiteRT (new)                                   | What it contains                                                                  |
| -------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `org.tensorflow:tensorflow-lite-support`     | `com.google.ai.edge.litert:litert-support`     | Support API **plus** the LiteRT runtime (`litert`)                                |
| `org.tensorflow:tensorflow-lite-support-api` | `com.google.ai.edge.litert:litert-support-api` | Support API only. The runtime comes separately (`litert-api` / `litert`).         |
| `org.tensorflow:tensorflow-lite-metadata`    | `com.google.ai.edge.litert:litert-metadata`    | `MetadataExtractor`, `MetadataParser`, and the metadata and model schema bindings |

## 2. Update Gradle dependencies

```diff
 dependencies {
-    implementation "org.tensorflow:tensorflow-lite-support:<old-version>"
-    implementation "org.tensorflow:tensorflow-lite-metadata:<old-version>"
+    implementation "com.google.ai.edge.litert:litert-support:<litert-version>"
+    implementation "com.google.ai.edge.litert:litert-metadata:<litert-version>"
 }
```

If you used the runtime-free `tensorflow-lite-support-api` because you provide
the runtime yourself:

```diff
-    implementation "org.tensorflow:tensorflow-lite-support-api:<old-version>"
-    implementation "org.tensorflow:tensorflow-lite:<old-version>"
+    implementation "com.google.ai.edge.litert:litert-support-api:<litert-version>"
+    implementation "com.google.ai.edge.litert:litert:<litert-version>"
```

> [!IMPORTANT]
> **Remove every `org.tensorflow:tensorflow-lite*` dependency.** The LiteRT
> artifacts keep the runtime classes (`org.tensorflow.lite.Interpreter`,
> `DataType`, etc.) and the TFLite model schema
> (`org.tensorflow.lite.schema.*`) under their original package names. If you
> mix old and new artifacts, the build fails with **duplicate class** errors.
> Run `./gradlew :app:dependencies` to find old artifacts that other libraries
> pull in transitively.

## 3. Update imports

Only the Support and Metadata packages move:

| Old package                                   | New package                                         |
| --------------------------------------------- | --------------------------------------------------- |
| `org.tensorflow.lite.support.*`               | `com.google.ai.edge.litert.support.*`               |
| `org.tensorflow.lite.support.metadata`        | `com.google.ai.edge.litert.support.metadata`        |
| `org.tensorflow.lite.support.metadata.schema` | `com.google.ai.edge.litert.support.metadata.schema` |

These packages keep their names:

-   `org.tensorflow.lite.*`: runtime API such as `Interpreter`,
    `InterpreterApi`, `DataType`, `Tensor`
-   `org.tensorflow.lite.schema.*`: TFLite model FlatBuffer schema

```diff
-import org.tensorflow.lite.support.image.TensorImage;
-import org.tensorflow.lite.support.image.ImageProcessor;
-import org.tensorflow.lite.support.image.ops.ResizeOp;
-import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
-import org.tensorflow.lite.support.metadata.MetadataExtractor;
+import com.google.ai.edge.litert.support.image.TensorImage;
+import com.google.ai.edge.litert.support.image.ImageProcessor;
+import com.google.ai.edge.litert.support.image.ops.ResizeOp;
+import com.google.ai.edge.litert.support.tensorbuffer.TensorBuffer;
+import com.google.ai.edge.litert.support.metadata.MetadataExtractor;
 import org.tensorflow.lite.DataType;       // unchanged
 import org.tensorflow.lite.Interpreter;    // unchanged
```

To rewrite imports in bulk in Java and Kotlin sources (GNU `sed`; on macOS, use
`sed -i ''`):

```shell
grep -rl 'org\.tensorflow\.lite\.support' --include='*.java' --include='*.kt' src/ \
  | xargs sed -i 's/org\.tensorflow\.lite\.support/com.google.ai.edge.litert.support/g'
```

Also update fully qualified names in ProGuard/R8 rules
(`-keep class org.tensorflow.lite.support.**`), reflection strings, and so on.

## 4. What stays the same

-   **Classes and APIs:** the same set of classes in `audio`, `common`, `image`,
    `label`, `model`, `tensorbuffer`, `text.tokenizers`, and `metadata`. Method
    signatures and behavior are unchanged.
-   **Metadata format:** models with TFLite metadata work unchanged. You don't
    need to re-convert or re-populate them.
-   **Minimum SDK:** `minSdkVersion` is still 19.

Internal-only changes you don't need to act on: nullness annotations moved to
type-use positions (for example `float @NonNull []`), `@CanIgnoreReturnValue`
was added to builder methods, and error messages are now formatted with
`Locale.ROOT`.

## 5. Build requirements

-   **JDK 17+:** the published artifacts contain Java 17 bytecode, so your
    Gradle build must compile with JDK 17 or newer (Android Gradle Plugin 8.x
    already requires this).
-   **FlatBuffers runtime:** `litert-metadata` depends on
    `com.google.flatbuffers:flatbuffers-java:25.2.10`. If your app pins an older
    `flatbuffers-java`, upgrade it to avoid runtime incompatibilities.

## 6. Out of scope

-   **TFLite Task Library** (`tensorflow-lite-task-vision`, `-text`, `-audio`):
    not part of this release.
-   **Python metadata writers / C++ metadata extractor:** this guide covers the
    Android Maven artifacts only.

## Checklist

-   [ ] Replace `org.tensorflow:tensorflow-lite-support[-api]` / `-metadata`
    with the `com.google.ai.edge.litert` equivalents
-   [ ] Remove the remaining `org.tensorflow:tensorflow-lite*` dependencies,
    including transitive ones
-   [ ] Rewrite `org.tensorflow.lite.support` →
    `com.google.ai.edge.litert.support` in code and R8/ProGuard rules
-   [ ] Build with JDK 17+
-   [ ] Rebuild, and run your inference and instrumentation tests
