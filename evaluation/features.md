1. Temporal Dependencies
In streaming data, each data point often depends on preceding observations, meaning the order of events directly influences the patterns present in the data. This sequential structure violates the i.i.d. assumption and requires models to capture relationships across time.

2. Long-Term Dependencies
Some patterns in a stream may depend on events that occurred far in the past, not just on recent history. Detecting such dependencies requires models to retain and utilize information from distant points, which is challenging for standard neural architectures.
**Event Streams**: Dependencies between process start and end or delayed effects across cases. Overlapping events because of concurrency (i.e., if you observe an event you dont know when its completes \<A_start, B_start, A_complete, B_complete \>)


3. Non-Linear Dependencies
Relationships between features or events in a stream are often complex and non-linear. Patterns may arise from intricate interactions across multiple attributes or time steps, making it harder for models to learn using simple or linear assumptions.
**Event Streams**: Non-linear dependencies manifest as complex routing, parallelism, or decision logic within processes, making pattern recognition over event streams more difficult.

4. Out-of-Order Events
In many streaming systems, data may arrive out of sequence due to network delays, buffering, or system asynchrony. This disrupts the true temporal order and can obscure or distort patterns that rely on proper event sequencing.
**Event Streams**: Out-of-order event arrival can result in incomplete or incorrect case reconstruction and affect the detection of process anomalies or compliance issues.

5. Fractal/Self-Similar Behavior
Streaming data may exhibit self-similar or fractal properties, where similar structures repeat at different scales or time windows. This multi-scale structure complicates pattern detection, as models must generalize across both fine-grained and coarse-grained behaviors.
**Event Streams**: Self-similarity means that process behaviors or deviations may recur at multiple levels (e.g., within cases, across cases, or over longer timeframes), making it harder to distinguish normal from anomalous behavior in process mining tasks.


temporal_dependency_strength: How much past events influence timing
long_term_dependency_strength: Long-term memory influence on decisions
non_linear_dependency_strength: Complex decision logic strength
out_of_order_strength: Probability and magnitude of delivery delays
fractal_behavior_strength: Fractal subprocess generation intensity

concurrency_percentage: Percentage of activities that can run concurrently