# Release 1.4.0

## LiteRT Next

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

### Breaking Changes

* `com.google.ai.edge.litert.TensorBufferRequirements`
  * It becomes a data class, so all fields could be accessed directly without getter methods.
  * The type of field `strides` changes from `IntArry` to `List<Int>` to be immutable.
* `com.google.ai.edge.litert.Layout`
  * The type of field `dimensions` and `strides` changes from `IntArry` to `List<Int>` to be immutable.

### Known Caveats

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>

### Major Features and Improvements

* <IF RELEASE CONTAINS MULTIPLE FEATURES FROM SAME AREA, GROUP THEM TOGETHER>

### Bug Fixes and Other Changes

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
