#ifndef ODML_LITERT_LITERT_GOOGLE_SAMPLE_APP_ANDROID_AOT_SRC_NATIVE_UTILS_H_
#define ODML_LITERT_LITERT_GOOGLE_SAMPLE_APP_ANDROID_AOT_SRC_NATIVE_UTILS_H_

#include <android/log.h>
#include <jni.h>

#include "util/java/jni_helper.h"

#define RETURN_IF_JNI_FAILURE(jni_helper)   \
  if ((jni_helper)->HasFailureOccurred()) { \
    return;                                 \
  }
#define RETURN_VAL_IF_JNI_FAILURE(jni_helper, ret_value) \
  if ((jni_helper)->HasFailureOccurred()) {              \
    return ret_value;                                    \
  }

// We use an extra level of macro indirection here to ensure that the
// macro arguments get evaluated, so that in a call to CHECK_CONDITION(foo),
// the call to STRINGIZE(condition) in the definition of the CHECK_CONDITION
// macro results in the string "foo" rather than the string "condition".
#define STRINGIZE(expression) STRINGIZE2(expression)
#define STRINGIZE2(expression) #expression

// Will log the result to Android's logcat.
#define CHECK_CONDITION(jni_helper, condition)                       \
  do {                                                               \
    ((condition)                                                     \
     ? CheckSucceeded(                                               \
           jni_helper, STRINGIZE(condition), __FILE__, __LINE__)     \
           : CheckFailed(jni_helper, STRINGIZE(condition), __FILE__, \
                                               __LINE__));           \
    RETURN_IF_JNI_FAILURE(jni_helper);                               \
  } while (false)
#define ASSERT_EQ(jni_helper, expected, actual) \
  CHECK_CONDITION(jni_helper, (expected) == (actual))
#define ASSERT_NE(jni_helper, expected, actual) \
  CHECK_CONDITION(jni_helper, (expected) != (actual))
#define ASSERT_STREQ(jni_helper, expected, actual) \
  ASSERT_EQ(jni_helper, 0, strcmp((expected), (actual)))

#define CHECK_CONDITION_RET(jni_helper, condition, ret_value)        \
  do {                                                               \
    ((condition)                                                     \
     ? CheckSucceeded(                                               \
           jni_helper, STRINGIZE(condition), __FILE__, __LINE__)     \
           : CheckFailed(jni_helper, STRINGIZE(condition), __FILE__, \
                                               __LINE__));           \
    RETURN_VAL_IF_JNI_FAILURE(jni_helper, ret_value);                \
  } while (false)
#define ASSERT_EQ_RET(jni_helper, expected, actual, ret_value) \
  CHECK_CONDITION_RET(jni_helper, (expected) == (actual), ret_value)
#define ASSERT_NE_RET(jni_helper, expected, actual, ret_value) \
  CHECK_CONDITION_RET(jni_helper, (expected) != (actual), ret_value)
#define ASSERT_STREQ_RET(jni_helper, expected, actual, ret_value) \
  ASSERT_EQ_RET(jni_helper, 0, strcmp((expected), (actual)), ret_value)

void CheckSucceeded(util::java::ThrowingJniHelper* jni_helper,
                    const char* expression, const char* filename,
                    int line_number) {
  __android_log_print(ANDROID_LOG_INFO, "TFLJNI",
                      "SUCCESS: CHECK passed: %s:%d: %s\n", filename,
                      line_number, expression);
}

void CheckFailed(util::java::ThrowingJniHelper* jni_helper,
                 const char* expression, const char* filename,
                 int line_number) {
  __android_log_print(ANDROID_LOG_ERROR, "TFLJNI",
                      "ERROR: CHECK failed: %s:%d: %s\n", filename, line_number,
                      expression);

  auto cls = jni_helper->FindClass("java/lang/IllegalStateException");
  jni_helper->ThrowNew(cls.get(), expression);
}

void LogToCallback(util::java::ThrowingJniHelper* jni_helper, jobject tfliteJni,
                   const char* messageFormat, ...) {
  // Setting up tfliteJni sendLogMessage invocations
  const char* callbackSignature = "(Ljava/lang/String;)V";
  util::java::ScopedLocalRef<jclass> callbackClass =
      jni_helper->GetObjectClass(tfliteJni);
  jmethodID printMethodId = jni_helper->GetMethodID(
      callbackClass.get(), "sendLogMessage", callbackSignature);

  // Format message
  char message[1024];
  va_list args;
  va_start(args, messageFormat);
  vsnprintf(message, sizeof(message), messageFormat, args);
  va_end(args);

  util::java::ScopedLocalRef<jstring> jmessage =
      jni_helper->NewStringUTF(message);
  jni_helper->CallVoidMethod(tfliteJni, printMethodId, jmessage.get());
}

#endif  // ODML_LITERT_LITERT_GOOGLE_SAMPLE_APP_ANDROID_AOT_SRC_NATIVE_UTILS_H_
