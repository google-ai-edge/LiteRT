"""Tests the availability of the model_utils under third_party/odml."""

from absl.testing import absltest as googletest


class ModelUtilsavailabilityTest(googletest.TestCase):
  """Tests the availability of the model_utils under third_party/odml."""

  def test_import_model_utils(self):
    """Verifies that the model_utils module and its submodules can be imported and accessed."""
    # pylint: disable=g-import-not-at-top
    import litert.python.tools.model_utils as mu
    from litert.python.tools.model_utils import core
    from litert.python.tools.model_utils.dialect import tfl
    # pylint: enable=g-import-not-at-top

    self.assertIsNotNone(mu)
    self.assertIsNotNone(core)
    self.assertIsNotNone(tfl)

    self.assertTrue(hasattr(mu, "core"))


if __name__ == "__main__":
  googletest.main()
