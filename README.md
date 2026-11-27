## Glicko-2 Tuning for Lichess

Research with Google Gemini, powered by https://huggingface.co/Lichess

### Background

Lichess ratings are useful for:
- match-making (lobby, puzzle, tournament)
- global ranking (leaderboard)
- TV and self-assessment (provisional - print question mark)

A properly configured Glicko-2 system offers great predictive accuracy
maximizing orthogonality of system characteristics, in stark contrast
to ad-hoc US Chess Rating system (age, half-K, bonus, effective N)
thresholds and piece-wise functions which are hard to tune, but also
in contrast to TrueSkill which can be sensitive to immediate outcomes.
This allows for testing evidence-based changes (first-move advantage).

### Lichess system parameters

	val minDeviation = 45
	val maxDeviation = 500d
	val defaultVolatility = 0.09d
	val maxVolatility = 0.1d
	val default = new Glicko(1500d, maxDeviation, defaultVolatility)

### References

- Glicko: https://glicko.net/glicko/glicko.pdf
- Glicko-2: https://glicko.net/glicko/glicko2.pdf
- Glicko-Boost: https://glicko.net/glicko/glicko-boost.pdf
