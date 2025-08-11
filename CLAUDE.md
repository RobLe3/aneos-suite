# aNEOS Project Configuration

## Development Framework
**MANDATORY**: All development tasks must follow the C&C + Implementation + Q&A agent structure defined in `DEVELOPMENT_FRAMEWORK.md`

## Default Agent Architecture
- **Command & Control (C&C)**: Task delegation and completion validation
- **Implementation**: Execute assigned tasks only  
- **Quality Assurance (Q&A)**: Independent verification of all work

## Key Rules
1. No self-validation allowed
2. All completion claims must pass Q&A verification
3. External ground truth required where possible
4. Maximum 3 iteration cycles per task

## Current Project Status
- Artificial NEO detection system exists but lacks ground truth validation
- Infrastructure is sophisticated but detection accuracy unproven
- Need real confirmed artificial vs natural object dataset for validation

## Priority
Establish baseline detection accuracy before further development.