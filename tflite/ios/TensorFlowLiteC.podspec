Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteC'
  s.version          = '2.14.0'
  s.summary          = 'TensorFlow Lite C library'
  s.description      = <<-DESC
  Internal TensorFlow Lite C library used by TensorFlowLiteSwift
  and TensorFlowLiteObjC. This pod should not be used directly.
  DESC

  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.license          = { :type => 'Apache' }
  s.authors          = 'Google Inc.'

  s.source           = {
    :http => "https://dl.google.com/tflite-release/ios/prod/tensorflow/lite/release/ios/release/30/20231002-210715/TensorFlowLiteC/2.14.0/883c6fc838e0354b/TensorFlowLiteC-2.14.0.tar.gz"
  }

  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '12.0'

  s.module_name = 'TensorFlowLiteC'
  s.library = 'c++'

  # Avoid unsupported simulator arch
  excluded_archs = 'i386'

  s.pod_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => excluded_archs
  }

  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => excluded_archs
  }

  s.default_subspec = 'Core'

  # ========================
  # Core Runtime
  # ========================
  s.subspec 'Core' do |core|
    core.vendored_frameworks = 'Frameworks/TensorFlowLiteC.xcframework'
  end

  # ========================
  # CoreML Delegate
  # ========================
  s.subspec 'CoreML' do |coreml|
    coreml.dependency 'TensorFlowLiteC/Core'
    coreml.weak_framework = 'CoreML'
    coreml.vendored_frameworks = 'Frameworks/TensorFlowLiteCCoreML.xcframework'
  end

  # ========================
  # Metal Delegate
  # ========================
  s.subspec 'Metal' do |metal|
    metal.dependency 'TensorFlowLiteC/Core'
    metal.weak_framework = 'Metal'
    metal.vendored_frameworks = 'Frameworks/TensorFlowLiteCMetal.xcframework'
  end
end
