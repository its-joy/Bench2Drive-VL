# Multiple Choice Questions — Per CDP Event
## Scenario: SignalizedJunctionRightTurn
## Goal: Turn right at a signalized intersection


---

## t=28.8 — Continue Straight + Accelerate (Junction Entry at Red Light)

### Q0 — Perception Check (V+G+H only)
```
The following frames show a sequence of what the ego vehicle 
just did. Look at the FIRST frame only to answer the following 
perception questions.

1. Are there any stop signs present in the scene?
   Answer yes or no only.

2. Are there any traffic lights present in the scene?
   Answer yes or no only.

3. If you answered yes to question 2, what colour is the
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 2, write N/A.

4. Are there any other vehicles visible in the scene,
   and if so, where are they relative to the ego vehicle?
   Select all that apply:
   - No other vehicles visible
   - Vehicle ahead
   - Vehicle behind
   - Vehicle to the left
   - Vehicle to the right
   - Vehicle in another position
```

**GT answers:** 1. yes | 2. no | 3. yes | 4. red 

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Accelerated straight through the junction
B) Turned right through the junction
C) Stopped before entering the junction
D) Changed lanes before entering the junction

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Think through each of the following steps before giving 
your final answer:

Step 1 — Location context:
Is the ego vehicle inside a junction or on a regular road segment? 
What visual cues support this?

Step 2 — Traffic control:
Is there a traffic light or stop sign affecting the ego vehicle? 

Step 3 — Action type:
What type of action did the ego vehicle perform?
Choose from: continue straight / turn right / turn left / 
lane change left / lane change right / accelerate / decelerate / stop
More than one may apply.

Step 4 — Speed:
Was the vehicle speeding up, slowing down, or maintaining speed?
What was the approximate speed at the start and end of this action?

Step 5 — Final description:
Summarise your observations from Steps 1-4 into a single coherent description of what the ego vehicle just did.
```
**GT anchor:** `continue_straight` + `accelerate`, speed 1.1→61.5 km/h

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle proceeded through 
the junction?

A) The ego proceeded despite a red traffic light
B) The ego proceeded because the traffic light had turned green
C) The ego proceeded because cross-traffic had cleared the junction
D) The ego proceeded because it was already committed and could not stop

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], explain why the ego vehicle took this action.

Think through each of the following steps before giving  your final answer:

Step 1 — Scene conditions:
What specific conditions are present in the scene?
Consider:
- Traffic signals and their state
- Road layout and lane markings
- Position and behaviour of nearby vehicles
- Any hazards or obstacles visible

Step 2 — Triggering evidence:
What specific evidence in the scene most directly relates 
to the action the ego vehicle just took?
Describe only what you can observe — do not infer intent.

Step 3 — Agent interaction:
Are there any vehicles, pedestrians, or obstacles that 
appear to have influenced this action?
If yes, describe their position, speed, and behaviour 
relative to the ego vehicle.

Step 4 — Final explanation:
Based only on your observations from Steps 1-3, explain 
why the ego vehicle took this action.
Do not guess intent — only reference what the evidence shows.
```
**GT anchor:** red light at 34.8m, cross-traffic left/ahead at 12.7m and 14.2m

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the ego had right of way and proceeded correctly
B) No — the ego violated a red traffic light
C) No — the ego created an unsafe situation by accelerating too fast
D) Partially — the ego was already committed to the junction

Answer key: B
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

Think through each of the following steps before giving 
your final answer:

Step 1 — Safety assessment:
Were there any other vehicles, pedestrians, or obstacles 
that posed a risk during this action?
If yes, describe their position, distance, and speed 
relative to the ego vehicle.
If no, state that the path was clear.

Step 2 — Traffic rule compliance:
Were any traffic rules relevant to this action?
Consider:
- Traffic signal state and whether it was obeyed
- Lane discipline and correct lane positioning
- Right of way
- Speed appropriateness for the situation
State which rules applied and whether each was followed or violated.

Step 3 — Outcome observation:
Did the action result in any observable consequence?
Consider:
- Contact or near-contact with another vehicle
- Sudden reactions from other road users
- Any abrupt changes in the ego vehicle's own behaviour

Step 4 — Final verdict:
Based only on your observations from Steps 1-3:
- State your verdict: safe / unsafe / partially safe
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule was violated
```
**GT anchor:** appropriate=False, ran_red_light, traffic_light=red at 34.8m

---
---

## t=47.4 — Lane Change Right + Decelerate (Junction, Collision)

### Q0 — Perception Check (V+G+H only)
```
Before answering any further questions, state the following 
from the frames provided:

1. Is the ego vehicle at or inside an intersection? 
   Answer yes or no only.

2. Are there any stop signs present in the scene? 
   Answer yes or no only.

3. Are there any traffic lights present in the scene? 
   Answer yes or no only.

4. If you answered yes to question 3, what colour is the 
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 3, write N/A.

```
**GT answers:*1. yes | 2. no | 3. no | 4. N/A

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Changed lanes to the right while slowing down
B) Changed lanes to the left while slowing down  
C) Maintained its lane while braking hard
D) Began turning right at the junction

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Include:
- The type of action performed
- The direction of any turn or lane change
- Whether the vehicle was speeding up or slowing down
- The approximate speed
```
**GT anchor:** `lane_change_right` + `decelerate`, speed 61.5→3.0 km/h

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle took this action?

A) The ego moved right to position into the correct lane for the intended right turn
B) The ego moved right to overtake a slower vehicle ahead
C) The ego braked due to a red traffic light ahead
D) The ego moved right to avoid an oncoming vehicle

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], 
explain why the ego vehicle took this action.

- Cite specific things you can observe in the scene
- Reference any relevant agents, signals, or road conditions
- Do not guess intent — only describe what the evidence shows
```
**GT anchor:** junction=True, speed 61.5 km/h, positioning for right turn

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the ego correctly positioned for the right turn
B) No — the ego violated a traffic signal during the lane change
C) No — the ego collided with another vehicle during the maneuver
D) Partially — the lane change was necessary but executed too quickly

Answer key: C
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

- State your verdict: yes / no / partially
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule
```
**GT anchor:** appropriate=False, collision with agent_68, insufficient deceleration

---
---

## t=62.2 — Lane Change Left (Post-Junction Repositioning)

### Q0 — Perception Check (V+G+H only)
```
Before answering any further questions, state the following 
from the frames provided:

1. Is the ego vehicle at or inside an intersection? 
   Answer yes or no only.

2. Are there any stop signs present in the scene? 
   Answer yes or no only.

3. Are there any traffic lights present in the scene? 
   Answer yes or no only.

4. If you answered yes to question 3, what colour is the 
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 3, write N/A.

```
**GT answers:*1. no | 2. no | 3. no | 4. N/A 

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Changed lanes to the left at low speed
B) Changed lanes to the right at low speed
C) Stopped and waited in the current lane
D) Turned left at an intersection

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Include:
- The type of action performed
- The direction of any turn or lane change
- Whether the vehicle was speeding up or slowing down
- The approximate speed
```
**GT anchor:** `lane_change_left`, speed 3.0→1.0 km/h

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle took this action?

A) The ego moved left to avoid an obstacle blocking its path ahead
B) The ego moved left to overtake a slower vehicle
C) The ego moved left to position for a left turn ahead
D) The ego moved left because its right lane was ending

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], 
explain why the ego vehicle took this action.

- Cite specific things you can observe in the scene
- Reference any relevant agents, signals, or road conditions
- Do not guess intent — only describe what the evidence shows
```
**GT anchor:** inferred_cause=obstacle_avoidance, speed 3.0 km/h post-junction
**Note:** inferred_cause may be `unknown` — accept any plausible observable explanation

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the lane change was performed safely at low speed
B) No — the ego changed lanes without checking for other vehicles
C) No — the ego should have stayed in the right lane for the turn
D) Partially — the lane change was safe but moved ego further from the turn lane

Answer key: A
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

- State your verdict: yes / no / partially
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule
```
**GT anchor:** appropriate=True, no infraction, low speed lane change

---
---

## t=80.2 — Lane Change Right + Accelerate (Unsafe Merge, Collision)

### Q0 — Perception Check (V+G+H only)
```
Before answering any further questions, state the following 
from the frames provided:

1. Is the ego vehicle at or inside an intersection? 
   Answer yes or no only.

2. Are there any stop signs present in the scene? 
   Answer yes or no only.

3. Are there any traffic lights present in the scene? 
   Answer yes or no only.

4. If you answered yes to question 3, what colour is the 
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 3, write N/A.

```
**GT answers:*1. no | 2. no | 3. yes | 4. green 

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Changed lanes to the right while accelerating
B) Changed lanes to the left while accelerating
C) Accelerated straight ahead in the same lane
D) Changed lanes to the right while braking

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Include:
- The type of action performed
- The direction of any turn or lane change
- Whether the vehicle was speeding up or slowing down
- The approximate speed
```
**GT anchor:** `lane_change_right` + `accelerate`, speed 1.0→31.7 km/h

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle took this action?

A) The ego moved right to position for the upcoming right turn
B) The ego moved right to avoid a vehicle that cut into its lane
C) The ego moved right because a traffic light ahead turned green
D) The ego moved right to overtake a slow vehicle blocking its path

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], 
explain why the ego vehicle took this action.

- Cite specific things you can observe in the scene
- Reference any relevant agents, signals, or road conditions
- Do not guess intent — only describe what the evidence shows
```
**GT anchor:** agent_112 at 4.7m right/behind at 18.7km/h, agent_108 right/ahead at 11.3m

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the ego successfully positioned into the right lane
B) No — the ego changed lanes without sufficient gap to approaching traffic
C) No — the ego violated a traffic signal during the lane change
D) Partially — the lane change was necessary but the acceleration was too sharp

Answer key: B
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

- State your verdict: yes / no / partially
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule
```
**GT anchor:** appropriate=False, collision with agent_112 at 0.3m, gap 4.7m insufficient

---
---

## t=83.8 — Lane Change Left + Decelerate (Emergency Correction)

### Q0 — Perception Check (V+G+H only)
```
Before answering any further questions, state the following 
from the frames provided:

1. Is the ego vehicle at or inside an intersection? 
   Answer yes or no only.

2. Are there any stop signs present in the scene? 
   Answer yes or no only.

3. Are there any traffic lights present in the scene? 
   Answer yes or no only.

4. If you answered yes to question 3, what colour is the 
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 3, write N/A.

```
**GT answers:*1. no | 2. no | 3. yes | 4. green 

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Changed lanes to the left while braking hard
B) Changed lanes to the right while braking hard
C) Braked hard while staying in the same lane
D) Changed lanes to the left while accelerating

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Include:
- The type of action performed
- The direction of any turn or lane change
- Whether the vehicle was speeding up or slowing down
- The approximate speed
```
**GT anchor:** `lane_change_left` + `decelerate`, speed 31.7→11.5 km/h, accel -5.611 m/s²

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle took this action?

A) The ego moved left and braked to avoid closing too fast on a vehicle directly ahead
B) The ego moved left to position for the upcoming right turn
C) The ego braked due to a red traffic light ahead
D) The ego moved left to overtake a slow vehicle on its right

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], 
explain why the ego vehicle took this action.

- Cite specific things you can observe in the scene
- Reference any relevant agents, signals, or road conditions
- Do not guess intent — only describe what the evidence shows
```
**GT anchor:** agent_108 at 6.3m aligned/ahead at 16.7km/h, ego at 31.7km/h closing fast

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the ego correctly avoided a collision with the vehicle ahead
B) No — the ego should have stayed in the right turn lane instead
C) No — the ego braked too hard and created a risk for following vehicles
D) Partially — the evasive action was necessary but the situation was self-caused

Answer key: D
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

- State your verdict: yes / no / partially
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule
```
**GT anchor:** appropriate=True, situation_avoidable=True caused by t=80.2
**Note:** accept both A and D — model identifying self-caused situation gets bonus credit

---
---

## t=85.0 — Turn Right (Target Maneuver, Wrong Lane, Yellow Light)

### Q0 — Perception Check (V+G+H only)
```
Before answering any further questions, state the following 
from the frames provided:

1. Is the ego vehicle at or inside an intersection? 
   Answer yes or no only.

2. Are there any stop signs present in the scene? 
   Answer yes or no only.

3. Are there any traffic lights present in the scene? 
   Answer yes or no only.

4. If you answered yes to question 3, what colour is the 
   traffic light? Select from: red / yellow / green / not visible.
   If you answered no to question 3, write N/A.

```
**GT answers:*1. yes | 2. no | 3. yes | 4. yellow 

---

### Q51 Easy — Describe the Action
```
Based on [the frames provided / the context provided], 
what action did the ego vehicle just perform?

A) Turned right at the junction while accelerating
B) Turned left at the junction
C) Continued straight through the junction
D) Stopped at the junction before turning

Answer key: A
```

### Q51 Hard — Describe the Action
```
Based on [the frames provided / the context provided], 
describe what the ego vehicle just did.

Include:
- The type of action performed
- The direction of any turn or lane change
- Whether the vehicle was speeding up or slowing down
- The approximate speed
```
**GT anchor:** `turn_right`, speed 11.5→27.2 km/h

---

### Q52 Easy — Explain Why
```
Based on [the frames provided / the context provided], 
what best explains why the ego vehicle took this action?

A) The ego executed the intended right turn at the signalized junction
B) The ego turned right to avoid a vehicle blocking the straight path
C) The ego turned right because the left lane was closed
D) The ego turned right to find an alternative route after missing the junction

Answer key: A
```

### Q52 Hard — Explain Why
```
Based on [the frames provided / the context provided], 
explain why the ego vehicle took this action.

- Cite specific things you can observe in the scene
- Reference any relevant agents, signals, or road conditions
- Do not guess intent — only describe what the evidence shows
```
**GT anchor:** junction=True, yellow light at 25.1m, goal=turn right

---

### Q53 Easy — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and appropriate?

A) Yes — the ego successfully completed the intended right turn
B) No — the ego turned right from the wrong lane
C) No — the ego should have stopped at the yellow light before turning
D) Partially — the turn was completed but from the wrong lane at a yellow light

Answer key: D
```

### Q53 Hard — Was This Appropriate
```
Based on [the frames provided / the context provided], 
was the action the ego vehicle just performed safe 
and compliant with traffic rules?

- State your verdict: yes / no / partially
- Cite the specific evidence that supports your verdict
- If unsafe, identify what made it unsafe
- If a traffic rule was violated, state which rule
```
**GT anchor:** appropriate=False, wrong_lane_right_turn, proceeded_at_yellow
**Note:** accept partially as correct — both wrong lane and yellow light must be cited for full marks

