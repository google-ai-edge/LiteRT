@testable import Sources
import XCTest

final class HelloWorldSwiftTests: XCTestCase {
    func testInit() {
        XCTAssertNotNil(BazelApp())
    }
}
