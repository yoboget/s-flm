"""Multi-mode denoiser (port of ModeFlowGraph's utils/modes.py).

With algo.num_modes = M > 1, the denoiser predicts a mixture over clean
sequences
  p(x0 | xt) = sum_m pi_m(xt) prod_l p_m(x0_l | xt),
i.e. a sequence-level mode head pi (B, M) and M sets of per-token
log-probs (B, L, M, V). Within a mode, tokens are independent; the mode
carries the dependencies between them.

Training (algo.hyp_loss):
  'mixture':  negative log-likelihood of the mixture.
  'wta':      winner-take-all, only the mode with the highest likelihood
              of x0 is trained on it, and the mode head learns which mode
              wins (commitment loss).
  'tempered': responsibilities r = posterior of the mixture (temperature
              1), held fixed, weight the likelihood of each mode, and the
              mode head is trained towards them.
Sampling (sampler.hyp_selection):
  'sample':  draw one mode per sequence from pi at every step.
  'average': per-token expectation of the mixture, sum_m pi_m p_m(x).
"""

import torch
import torch.nn.functional as F


HYP_LOSSES = ('mixture', 'wta', 'tempered')


def mode_nll(mode_log_probs, log_probs, x0, weight, valid_tokens,
             hyp_loss):
  """Multi-mode loss, spread over the tokens of each sequence.

  Args:
    mode_log_probs: (B, M) log pi.
    log_probs: (B, L, M, V) per-mode log p_m(x0_l | xt).
    x0: (B, L) clean tokens.
    weight: (B, L) tokens entering the likelihood (e.g. only the masked
      ones for absorbing diffusion).
    valid_tokens: (B, L) tokens over which the loss is averaged.
  Returns:
    (B, L) per-token loss whose sum over a sequence is its loss, so that
    trainer_base._loss gives sum_b loss_b / num_tokens.
  """
  M = log_probs.shape[2]
  token_lp = log_probs.gather(
    -1, x0[:, :, None, None].expand(-1, -1, M, 1)).squeeze(-1)
  seq_lp = (token_lp * weight[..., None]).sum(1)  # (B, M)
  joint_lp = mode_log_probs + seq_lp
  # Sequences without any element in the likelihood (absorbing
  # diffusion at low noise) carry no information on the mode.
  has_elements = (weight.sum(-1) > 0).to(seq_lp.dtype)
  num_tokens = valid_tokens.sum(-1).to(seq_lp.dtype)

  if hyp_loss == 'mixture':
    seq_loss = -torch.logsumexp(joint_lp, -1)
  elif hyp_loss == 'wta':
    best = seq_lp.argmax(-1)
    recon = -seq_lp.gather(-1, best[:, None]).squeeze(-1)
    commit = F.nll_loss(mode_log_probs, best, reduction='none')
    seq_loss = recon + has_elements * num_tokens * commit
  elif hyp_loss == 'tempered':
    resp = joint_lp.softmax(-1).detach()
    recon = -(resp * seq_lp).sum(-1)
    commit = -(resp * mode_log_probs).sum(-1)
    seq_loss = recon + has_elements * num_tokens * commit
  else:
    raise ValueError(
      f'Unknown hyp_loss {hyp_loss}. Available: {HYP_LOSSES}')

  per_token = seq_loss / num_tokens.clamp(min=1)
  return per_token[:, None] * valid_tokens


def select_mode(mode_log_probs, log_probs, hyp_selection):
  """Turns the multi-mode prediction into per-token log-probs (B, L, V),
  by sampling one mode per sequence or taking the expectation of the
  mixture."""
  if hyp_selection == 'average':
    return torch.logsumexp(
      mode_log_probs[:, None, :, None] + log_probs, dim=2)
  if hyp_selection == 'sample':
    m = torch.multinomial(mode_log_probs.exp().float(), 1).squeeze(-1)
    b = torch.arange(len(m), device=m.device)
    return log_probs[b, :, m]
  raise ValueError(f'Unknown hyp_selection {hyp_selection}. '
                   'Available: sample, average')
