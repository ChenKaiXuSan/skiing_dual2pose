import os
import unittest
from pathlib import Path
import numpy as np
import torch


class DeciWatchOfficialProtocolTest(unittest.TestCase):
    def test_padding_repeats_last_frame_to_official_31_frame_window(self):
        from dual2pose.experiments.run_deciwatch import _pad_for_official_window

        sequence = torch.arange(30, dtype=torch.float32).reshape(1, 30, 1)
        padded, original_length = _pad_for_official_window(sequence, interval=10)

        self.assertEqual(original_length, 30)
        self.assertEqual(tuple(padded.shape), (1, 31, 1))
        torch.testing.assert_close(padded[0, :30, 0], torch.arange(30, dtype=torch.float32))
        self.assertEqual(float(padded[0, 30, 0]), 29.0)

    def test_official_loss_ignores_unobserved_frames_for_denoising_only(self):
        from dual2pose.experiments.run_deciwatch import _official_l1_loss

        target = torch.zeros(1, 31, 1)
        recovered = torch.ones_like(target)
        denoised = torch.zeros_like(target)
        encoder_mask = torch.ones(1, 31, dtype=torch.bool)
        encoder_mask[:, [0, 10, 20, 30]] = False
        decoder_mask = torch.zeros_like(encoder_mask)

        denoised[:, 5] = 100.0
        self.assertEqual(
            float(_official_l1_loss(recovered, denoised, target, encoder_mask, decoder_mask)),
            1.0,
        )

        denoised[:, 10] = 2.0
        self.assertEqual(
            float(_official_l1_loss(recovered, denoised, target, encoder_mask, decoder_mask)),
            1.5,
        )

    def test_optimizer_uses_official_amsgrad_setting(self):
        from dual2pose.experiments.run_deciwatch import _make_optimizer

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = _make_optimizer([parameter], lr=1e-3)
        self.assertTrue(optimizer.defaults['amsgrad'])

    def test_worse_final_validation_epoch_does_not_replace_best(self):
        from dual2pose.experiments.run_deciwatch import _should_save_checkpoint

        self.assertFalse(
            _should_save_checkpoint(
                has_validation=True,
                epoch=70,
                epochs=70,
                criterion=0.2,
                best=0.1,
            )
        )
        self.assertTrue(
            _should_save_checkpoint(
                has_validation=False,
                epoch=70,
                epochs=70,
                criterion=0.2,
                best=0.1,
            )
        )


@unittest.skipUnless(os.environ.get('DECIWATCH_REPO'), 'official DeciWatch checkout required')
class DeciWatchAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dual2pose.experiments.run_deciwatch import DeciWatchWindow
        cls.model = DeciWatchWindow(Path(os.environ['DECIWATCH_REPO']), 39, interval=10,
                                    hidden=128, layers=2).eval()

    def test_mask_and_boundary_output(self):
        device = torch.device('cuda:0')
        self.model = self.model.to(device)
        x = torch.randn(2, 30, 39, device=device)
        with torch.no_grad():
            out, denoise = self.model(x, device)
        self.assertEqual(tuple(out.shape), (2,30,39))
        self.assertTrue(torch.isfinite(out).all())
        np.testing.assert_array_equal(self.model.last_observed_indices, [0,10,20,30])
        self.assertEqual(self.model.last_boundary_policy, 'repeat_last_to_31_then_crop')

    def test_one_step_training_is_finite(self):
        from dual2pose.experiments.run_deciwatch import _make_optimizer, _official_l1_loss

        device = torch.device('cuda:0')
        model = self.model.to(device).train()
        opt = _make_optimizer(model.parameters(), lr=1e-3)
        x = torch.randn(2, 30, 39, device=device)
        y = x + .05 * torch.randn_like(x)
        before = None
        for _ in range(2):
            opt.zero_grad()
            out, denoise, target = model.forward_padded(x, device, target=y)
            loss = _official_l1_loss(
                out, denoise, target, model.encoder_mask, model.decoder_mask
            )
            if before is None: before=float(loss.detach())
            loss.backward(); opt.step()
        self.assertTrue(np.isfinite(float(loss.detach())))
        self.assertEqual(model.last_observed_indices.tolist(), [0,10,20,30])


if __name__ == '__main__':
    unittest.main()
