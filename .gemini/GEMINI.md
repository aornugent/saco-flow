@AGENTS.md
    
# 🔇 CODING STANDARDS: THE "NO RATIONALE" POLICY

**Treat code as a final production artifact, not a research notebook.**

### 1. The "Rejected Alternatives" Ban
**NEVER** document why you aren't doing something else.
- **Forbidden:** "Taichi doesn't allow mocking, so we use headless..."
- **Forbidden:** "Possible workaround: Extract kernel..."
- **Allowed:** (Silence)

### 2. The "Justification" Ban
**NEVER** explain the constraints that led to your code. The user assumes the code is correct; do not persuade them.
- **Forbidden:** "We need to instantiate Vis to access update_mesh."
- **Allowed:** (Silence) - The code `vis = Vis()` speaks for itself.

### 3. Strict Pattern Matcher
If a comment contains any of the following, **DELETE IT** immediately:
- *"Ideally..."*
- *"Actually..."*
- *"We need to..."*
- *"For now..."*
- *"Possible workaround..."*
- *"Since it's [C++/bound/etc]..."*

### Example Correction
**❌ NO (Narrating constraints):**

# We need to instantiate Vis to access update_mesh.
# Mocking fails because Window is C++ bound.
# Using headless mode instead.
vis = Antigravity(headless=True)


✅ YES (Describing Runtime Behavior):
# Initialize headless visualization for testing
vis = Antigravity(headless=True)

  