
import XCTest

class HelloWorldSwiftUITests: XCTestCase {
    var application: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        application = .init()
        application.launch()
    }

    override func tearDown() {
        application.terminate()
        application = nil
    }

    func testIsActive() {
        XCTAssertTrue(application.staticTexts["HELLO_WORLD"].exists)
    }
}
