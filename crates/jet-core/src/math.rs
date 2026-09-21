use crate::{JetError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedScore {
    pub log_probability: f64,
    pub normalized_log_probability: f64,
    pub normalized_probability: f64,
}

pub fn normalize_log_probabilities(values: &[f64]) -> Result<Vec<NormalizedScore>> {
    if values.is_empty() {
        return Err(JetError::InvalidRequest(
            "at least one candidate is required".to_owned(),
        ));
    }
    if values
        .iter()
        .any(|value| value.is_nan() || *value == f64::INFINITY)
    {
        return Err(JetError::NonFiniteScore(
            "NaN and positive infinity are not valid log-probabilities".to_owned(),
        ));
    }

    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return Err(JetError::NonFiniteScore(
            "all candidates have negative-infinite log-probability".to_owned(),
        ));
    }

    let sum_exp: f64 = values.iter().map(|value| (*value - max).exp()).sum();
    let log_normalizer = max + sum_exp.ln();

    Ok(values
        .iter()
        .map(|value| {
            let normalized_log_probability = *value - log_normalizer;
            NormalizedScore {
                log_probability: *value,
                normalized_log_probability,
                normalized_probability: normalized_log_probability.exp(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_extreme_values_stably() -> Result<()> {
        let scores = normalize_log_probabilities(&[-10_000.0, -10_001.0])?;
        let sum: f64 = scores
            .iter()
            .map(|score| score.normalized_probability)
            .sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(scores[0].normalized_probability > scores[1].normalized_probability);
        Ok(())
    }

    #[test]
    fn rejects_all_negative_infinity() {
        assert!(normalize_log_probabilities(&[f64::NEG_INFINITY]).is_err());
    }

    #[test]
    fn preserves_impossible_candidates_as_zero_probability() -> Result<()> {
        let scores = normalize_log_probabilities(&[0.0, f64::NEG_INFINITY])?;
        assert_eq!(scores[0].normalized_probability, 1.0);
        assert_eq!(scores[1].normalized_probability, 0.0);
        Ok(())
    }

    #[test]
    fn rejects_nan_and_positive_infinity() {
        assert!(normalize_log_probabilities(&[f64::NAN, 0.0]).is_err());
        assert!(normalize_log_probabilities(&[f64::INFINITY, 0.0]).is_err());
    }
}
