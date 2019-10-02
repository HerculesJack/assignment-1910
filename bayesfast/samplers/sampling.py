import numpy as np

    """def warnings(self):
        # list.copy() is not available in python2
        warnings = self._warnings[:]

        # Generate a global warning for divergences
        message = ""
        n_divs = self._num_divs_sample
        if n_divs and self._samples_after_tune == n_divs:
            message = (
                "The chain contains only diverging samples. The model "
                "is probably misspecified."
            )
        elif n_divs == 1:
            message = (
                "There was 1 divergence after tuning. Increase "
                "`target_accept` or reparameterize."
            )
        elif n_divs > 1:
            message = (
                "There were %s divergences after tuning. Increase "
                "`target_accept` or reparameterize." % n_divs
            )

        if message:
            warning = SamplerWarning(
                WarningType.DIVERGENCES, message, "error", None, None, None
            )
            warnings.append(warning)

        warnings.extend(self.step_adapt.warnings())
        return warnings"""
        