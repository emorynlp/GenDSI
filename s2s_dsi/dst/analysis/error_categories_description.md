### Non-existing slots: the information that is not explicitely stated in the turn and cannot be inferred from the turn.

> The following example would be considered as a case for 1 non-existing slot, 1 good slot, and 1 missing information:

Turn: Okay, we'll definitely need to avoid peanuts and lower your sodium intake to manage your blood pressure. Are you currently physically active?

Extractions:

`"is_alcohol_served": ['?']`
`"activity_level": ['?']`

In this example, the slot "activity_level" is good because whether the person is physically active is being asked. The slot "is_alcohol_served" is an non-existing slot because it is not present in the turn. The first sentence is completely omitted. In the best case scenario, we can summarize this first sentence into 1 slot-value pair, being `"actions": ['avoid peanuts', 'lower sodium intake', 'manage blood pressure']`. Therefore it is considered as 1 missing information.

---

### False information: wrong extraction that is against the content of the turn.

> The following example would be considered as a case for 2 false information slots: 

Turn: Sure, let's have a sectional rehearsal for that. Bass and cello sections, please come forward. Great job, everyone. I noticed that in measure 100, some of you were playing a bit too loudly. Please remember to keep the dynamics level until measure 112.

Extractions:

`"beginning_note_of_song": ['100']`
`"end_of_song": ['112']`

---

### Incomplete information: incomplete extraction of information that is explicitely stated in the turn.

> The following example would be considered as a case for an incomplete information slot:

Turn: I believe the target audience for the yoga article would be pregnant women and new mothers, while the sugar article would appeal to a broader group interested in health and wellness.

Extractions:

`"audience_type": ['Health & Wellness']`

---

### Hallucination: a type of non-existing slots. The information is clearly directly from one or more previous turns.

> The following example would be considered as a case for 2 hallucinations and 1 good slot:

Turn: Alright, Sarah. Can you tell me which payroll period the issues occurred in?

Extractions:

`"name": ['Sarah Johnson']`
`"employee iD": ['123456']`
`"payroll period": ['?']`

---

### Redundant Slots: extractions that overlaps with other extractions

> The following example would be considered as a case for 3 redundant slots and 2 good slots:

Turn: My name is John Smith and you can reach me at 555-1234.

Extractions:

`"name": ['John Smith']`
`"contact info": ['555-1234']`
`"first name": ['John']`
`"last name": ['Smith']`
`"phone number": ['555-1234']`

---

### Vague: the generated information is correct, but is too vague to be actually used.

> The following example would be considered as a case of 1 good slot and 3 vague slots:

Turn: When I was growing up, my parents always taught me to appreciate nature and not take it for granted. We used to go on camping trips and nature walks, and those memories stay with me even today.

Extractions:

`"parents teachings": ['Appreciate nature']`
`"camping trips": ['Yes']`
`"nature walks": ['Yes']`
`"camping nature memories": ['Yes']`

---


### Missing information: the information that is explicitely stated in the turn and should be indisputably appear in the extraction but is not.

> The following example would be considered as a case for 1 false information, 1 good slot, and 2 missing information:

Turn: Well, my current schedule is Monday through Friday, 8:00 am to 5:00 pm. I was thinking about changing it up a bit.

Extractions: 

`"day_of_appointment": ['Monday']`
`"end_of_the_available_event": ['5:00 pm']`

In this example, the "end_of_the_available_event" slot is good although not ideal. "day_of_appointment" slot is false information because it is not an appointment. The missing information could be summarized into 2 slots, being `"start_time": ['8:00 am']` and `"current_schedule": ['Monday through Friday']`. It would be difficult to summarize the information into 1 slot without trampling the good slot that is generated by the model, and therefore considered as 2 missing information.

---

### No-extraction turns: turns with no extraction.

