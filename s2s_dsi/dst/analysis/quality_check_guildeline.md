# Quality Check Guideline

## Task Overview
The task involves reviewing dialogues turn-by-turn to assess the correctness of filled slots, identifying missing information, and ensuring the reasonability of unfilled slots. The goal is to ensure that the information in each dialogue turn is accurately represented in its slots.

## 1. Checking Filled Slots
   - **Slot Names**:
     - Slot names should describe a type of information, not a specific value.
     - Ensure that slot names align with the information present in the turn.
     - Modify slot names if they are too specific or not reflective of the information.
     - If a slot name cannot be fixed, remove the slot.
   - **Slot Values**:
     - Slot values should be mentioned or strongly implied in the turn.
     - Verify that slot values match the corresponding slot names.
     - Modify slot values if they do not align with the information.
     - If a slot value cannot be fixed, remove the slot.
   - **Slot Descriptions**:
     - Slot descriptions should reasonably represent the slot name and value.
     - Ensure that slot descriptions do not explicitly reveal the slot value.
     - Modify slot descriptions if they give away the slot value.

## 2. Adding Missing Information
   - Check if there is any information that is missing but explicitly or coreferentially mentioned in the turn.
   - Only add a slot if you are confident that the slot is correct and directly related to the turn's content.
   - Add missing slots by creating a new row below the existing filled slots (above the unfilled slots).

## 3. Reviewing Unfilled Slots
   - Unfilled slots should describe information NOT present in the turn.
   - Unfilled slots should be loosely related to the topic of conversation.
   - No need to assess the reasonability of unfilled slot names or descriptions.

## Guidelines for Editing The CSV

### Slot Modification
   - To modify a slot, separate the existing value with two pipes (||) and add the new value after the pipe(s). Example: "Old Value || New Value"

### Slot Removal
   - If a slot name, value, or description cannot be fixed, delete the row containing that slot.

### Adding Missing Slots
   - Ensure that missing slots have a name, value, and description.
   - Add missing slots in a new row below the existing filled slots, above unfilled slots.

### Example Workflow
   1. Review the turn's content.
   2. Check if filled slots are correct (name, value, description).
   3. Modify or remove slots as needed.
   4. Check for missing information and add slots if appropriate.
   5. Review unfilled slots for relevance.

## Additional Notes
- Maintain consistency in slot naming conventions.
- Focus on accuracy, clarity, and concision in slot descriptions.
- Avoid introducing new information or making assumptions not supported by the dialogue.
- If a turn does not contain any relevant information for slot filling, no action is required for that turn.