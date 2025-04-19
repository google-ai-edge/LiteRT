package com.google.ai.edge.litert.deployment

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.ModelProvider
import com.google.android.play.core.aipacks.AiPackManagerFactory
import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.AiPackStateUpdateListener
import com.google.android.play.core.aipacks.model.AiPackStatus
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * A [ModelProvider] that's backed by on-demand Google Play AI Pack.
 *
 * To use this class, you need to declare the following permission in your manifest:
 * ```xml
 * <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
 * ```
 *
 * NOTE: For install-time AiPack, use [ModelProvider.staticModel] instead.
 */
class AiPackModelProvider(
  private val context: Context,
  private val aiPackName: String,
  private val modelPath: String,
  private val accelerator: Accelerator,
  private vararg val moreAccelerators: Accelerator,
) : ModelProvider {
  private val aiPackManager = AiPackManagerFactory.getInstance(context)

  /** It's always a file, for on-demand AiPack. */
  override fun getType() = ModelProvider.Type.FILE

  override fun isReady(): Boolean {
    val location = aiPackManager.getPackLocation(aiPackName)
    return (location != null).also { Log.i(TAG, "AiPack $aiPackName is installed = $it") }
  }

  /**
   * Returns the absolute path to the model file.
   *
   * @throws IllegalStateException if the AiPack is not ready yet.
   */
  override fun getPath(): String {
    if (!isReady()) {
      throw IllegalStateException("AiPack is not ready yet")
    }
    return aiPackManager.getAiAssetLocation(aiPackName, modelPath)!!.path()
  }

  override fun getCompatibleAccelerators() = setOf(accelerator, *moreAccelerators)

  override suspend fun download() {
    if (!isReady()) {
      suspendCancellableCoroutine { continuation ->
        val listener =
          object : AiPackStateUpdateListener {
            override fun onStateUpdate(state: AiPackState) {
              if (state.name() == aiPackName) {
                Log.i(TAG, "AiPack $aiPackName status = ${state.status()}")
                when (state.status()) {
                  AiPackStatus.COMPLETED -> unregisterListenerAndResume(continuation, this)
                  AiPackStatus.CANCELED -> {
                    Log.w(TAG, "AiPack $aiPackName download is canceled")
                    unregisterListenerAndResume(
                      continuation,
                      this,
                      IllegalStateException("AiPack download is canceled"),
                    )
                  }
                  AiPackStatus.FAILED -> {
                    Log.e(
                      TAG,
                      "AiPack $aiPackName failed to download, errorCode = ${state.errorCode()}",
                    )
                    unregisterListenerAndResume(
                      continuation,
                      this,
                      IllegalStateException("AiPack failed to download"),
                    )
                  }
                  AiPackStatus.WAITING_FOR_WIFI,
                  AiPackStatus.REQUIRES_USER_CONFIRMATION -> {
                    // TODO(niuchl): Show a dialog to ask user to confirm for large on
                    // cellular data.
                    unregisterListenerAndResume(
                      continuation,
                      this,
                      IllegalStateException("Waiting for user confirmation"),
                    )
                  }
                  AiPackStatus.UNKNOWN -> {
                    Log.e(TAG, "Asset pack status unknown")
                    unregisterListenerAndResume(
                      continuation,
                      this,
                      IllegalStateException("Asset pack status unknown"),
                    )
                  }
                  else -> {
                    Log.i(TAG, "AiPack $aiPackName is downloading")
                  }
                }
              }
            }
          }

        aiPackManager.registerListener(listener)
        aiPackManager.fetch(listOf(aiPackName)).addOnFailureListener {
          unregisterListenerAndResume(continuation, listener, it)
        }
        continuation.invokeOnCancellation { aiPackManager.unregisterListener(listener) }
      }
    }
  }

  private fun unregisterListenerAndResume(
    continuation: CancellableContinuation<Unit>,
    listener: AiPackStateUpdateListener,
    exception: Exception? = null,
  ) {
    aiPackManager.unregisterListener(listener)
    if (exception != null) {
      continuation.resumeWithException(exception)
    } else {
      continuation.resume(Unit)
    }
  }

  companion object {
    private const val TAG = "AiPackModelProvider"
  }
}
