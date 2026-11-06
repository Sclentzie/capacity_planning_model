# Build log

This repository is published as a single commit, so this file is the
record of how it was actually built: six working sessions between 20
October and 6 November 2026, originally 54 commits. The dates are real and
the order is the order things happened in.

It is kept because the interesting part of a model is not the finished
code. It is which assumptions turned out to be wrong, what the tests
caught, and what had to be withdrawn after it had already been reported.
Several figures in this project were published to the architect, found to
be wrong, and corrected. Those reversals are written down here rather than
quietly edited out.

Each session worked to the specification in
[Capacity_Model_Spec.md](Capacity_Model_Spec.md), one task at a time, with
the architect approving each task before the next began.

---

## Session 1 — the specification (20–21 October)

No code. Every input, formula, default and validation check was agreed and
written down first, including the things the model would deliberately not
do. The specification stayed the source of truth for the whole build: when
code and spec disagreed, the code was wrong, and when the spec itself
turned out to be wrong it was changed on the record with a date and a
reason.

## Session 2 — demand and the forecast (21 October, 7 commits)

**Scaffolding and inputs.** Project layout, the three YAML input files,
and a smoke test that loads them. From the start no number lives in the
code.

**Synthetic history.** 156 weeks for six sites, built from named
ingredients — site size, growth, seasonality, holiday effects, and one
noise draw — with tests that recover each ingredient back out of the
generated data. Requests are kept as fractional expected values rather
than rounded, because rounding 10,296 cells would invent or lose work
every week.

**The forecast.** Classical decomposition and Holt-Winters, both fitted on
all but the last 13 weeks and graded on those 13. The seasonal index is
keyed to position in a 52-week cycle rather than the ISO week number,
because 2026 is a 53-week year and an ISO key would have shifted every
seasonal factor partway through the history.

**Holiday effects are divided out before fitting** and multiplied back on
by date. A unit test on a clean series caught the seasonal index drifting
by 0.19% — the data itself had not shown it. Dividing an event out and
multiplying it back cancels exactly only if both steps use the same basis,
and the test is what proved they did.

**Requests to work volume.** One forecast drives 24 work types, each
measured per request or per chart and each carrying a lag that places its
hours in the week the work lands. Work lagged past the final plan week is
written to a conservation file rather than dropped, so totals still tie
out; the worst conservation difference across all work-type and medium
combinations is reported rather than assumed to be zero.

## Session 3 — workload (22 October, 9 commits)

**Erlang C** for the phones and the walk-in counter, checked against a
reference case worked out by hand in Excel before the code existed
(`docs/validation/erlang_hand_check.xlsx`). The hand calculation came
first and the code was written to meet it.

**Intraday shape.** Half-hour curves for arrivals, then break and lunch
coverage placed by load rather than evenly. An early version could produce
an impossible relief pattern; that case is now caught by a test rather
than by inspection.

**A blended onsite team.** The counter and the phones are served by the
same people, so they are solved together at the stricter of the two goals
and then topped up until the other holds. Site email is absorbed into idle
time rather than staffed separately.

**The back-office backlog.** Queues carry what the week before left and
are measured in days of inventory against stage goals, checked against a
four-week roll worked by hand in Excel
(`docs/validation/backlog_hand_roll.xlsx`).

**A duplication removed before it could bite.** Retrieval times had been
living in two tables at once. Both said 12 minutes for digital, so nothing
was visibly wrong — which is exactly how that kind of defect survives
until somebody retunes one side and not the other. The media table is now
the only place those times live, and the code raises if a time creeps back
into the other.

**Risk-based quality review.** Full coverage for the request types that
need it plus a sample of the rest, with the redaction share *read* from
the redaction work type rather than repeated, so retuning one retunes the
other. A test changes redaction to 50% and checks the review coverage
follows.

## Session 4 — supply, coverage, hiring and cost (26–27 October, 12 commits)

**The supply ledger.** Headcount, hires, attrition, the training curve and
shrinkage, week by week and role by role. Headcount stays fractional,
because attrition is a rate applied to a headcount and not a list of named
leavers. No person-level identifiers are stored anywhere in the model.

**Two of the builder's own errors, caught by tests.** A compounded figure
that should have been additive, and a hand-computed constant wrong in the
fourth decimal place. Both were found by the tests rather than by review.

**Roster tuning.** The opening roster was tuned to the measured
requirement. Two mistakes here were visible only in intermediate values —
they set the roster wrongly without changing anything in the printed
output — which is the argument for testing intermediate values at all.

**The gap, and what follows from it.** Back-office queues are checked by
week; the counter is checked by day, because a walk-in cannot wait until
Thursday.

**Coverage events.** Borrow from a nearby site first, then overtime
against a cap, then accept the gap — each written to a log with the fields
a real coverage log carries. The log's *shape* is treated as a contract:
it must have every column whether or not a given run produced that kind of
event.

**Hiring.** A role that stays more than 1 FTE short for three consecutive
weeks triggers a class, sized to the gap and rounded up to whole people,
with a hire-by date counted back through requisition, onboarding and ramp.
Overtime bridges short dips and new hires' ramp-up; it is not the staffing
plan.

**plan.py split** into plan, coverage and hiring. A reorganisation with no
behaviour in it — and the only way to say that with a straight face was to
prove it: all 18 generated files were byte-identical afterwards.

**Automation and human-in-the-loop work.** A machine does some of the work
and hands some back: exceptions, a QA sample, and legal sign-off. Each
term is priced at the rate of whoever actually does it — the QA sample
goes to the QA team, whose hour costs more than a Tier 1 hour, so pricing
it at the owning role's rate would have flattered automation on every work
type QA does not own.

**A defect the change itself found.** Fixed costs defaulted to zero for a
work type the machine never touched, which put the two work types the
specification *pins to 0%* at the top of the savings ranking — at 61, 47
and 33 times return, on the strength of a per-item fee they can never
incur. Not measurable is not the same as free. Those rows now carry no
after-fixed figures at all and sort last: reported, not recommended.

**Both sides must cover the same work.** Any automation-against-by-hand
comparison now carries the human-in-the-loop labour and the platform fee.
On one work type in one week that moved the comparison from $278 against
$664 to $444 against $664 — the exception and QA labour was a quarter of
automation's true cost, and leaving it out compared a machine that never
needs a second look against a person who does.

## Session 5 — scenarios (27 October – 3 November, 18 commits)

Twenty-one runs built up over seven working days. No scenario re-tunes its
roster, so each reports what the change costs the operation it actually
has.

**The scenario engine**, then volume and attrition shocks, phone service
goals, lunch-window and Monday scheduling shapes, a single-site shortfall,
pooled phones, five automation variants, a central records archive,
coordinated digitization, a proof-of-concept gate, and combinations.

**A sweep instead of a guess.** The part-time Monday share was swept from
0% to 10% in one-point steps rather than asserted: 2% is best on cost, 4%
breaks even, 7% removes the most uncovered hours at about $75 each, and
10% is worse than the baseline on both measures.

**The handle-time uplift was split in two.** A flat +10% uplift turned out
to be impossible as a mix effect — removing easy work cannot raise total
workload, and at the ceiling it can only hold it flat. So the calculated
mix uplift and a separate escalation input are now counted apart, and the
break-even escalation is reported rather than assumed.

**Easy-item valuation.** The items a machine takes are the simple ones, so
what it displaces is what a *simple* item would have cost by hand, not an
average one. That single decision cut the list of work types that pay back
from six to two.

### Three defects that inverted published findings

**The turnaround filter that never bit.** A test had been written for a
filter that removed human-in-the-loop lines from the turnaround
calculation. The test passed — on a frame that never carried those lines
in the first place, so it could not have failed. The test now asserts the
filter *bites*, rather than only that removing the lines by hand changes
nothing.

**A queue key that collided.** Queues were keyed by role, scope, work type
and medium. With automation on, the human line and the exception line for
the same work type sit on the same owning role, so they collided and the
later row silently replaced the earlier one. Whole human queues were
vanishing from the projection in both automation runs.

Those two together had inverted a headline. A turnaround *improvement*
from 4.67 to 4.01 days had already been reported. Counting each item once
and projecting every queue, the real figure was 4.67 to 4.90 — the
programme made turnaround **worse**, with weeks over goal going from 6 to
19.

**A queue that silently restarted.** Chasing why the half-realized
automation run showed worse turnaround than both the baseline and the full
run turned up a third defect. The queue key still included the handle
time, which was harmless until handle times began moving with the ramp.
From then on a queue got a new key whenever its handle time changed, the
lookup missed, and the queue **restarted at its target backlog with its
real position thrown away** — no error, no warning.

| | broken week-to-week rolls, of 3,519 |
|---|---|
| Baseline | 0 (handle times never move) |
| Automation at 100% | 196 |
| Automation at 50% | 820 |

Which is exactly why the half-realized run looked worst of all. Hiring
timing, back-office shortfall and hour distribution were each ruled out
first. **Two earlier findings were withdrawn** as a result: "half a
programme is worse than none", and a claimed go-live ramp penalty of
0.136 days. Both were this bug.

### How those fixes were proven

Both fixes were then **proven by mutation**: the fix was reverted and the
tests re-run to confirm they fail without it. Reverting the turnaround
filter fails two named tests; reverting the queue key fails two more,
including one that counts the queue lines going in against the lines
coming out. The mutations were then reverted by `git checkout` and all
four tests pass again.

This mattered because of the first defect above. A test that has never
failed is a test nobody has checked.

### Other corrections in Session 5

**The central archive scenario reported an artefact.** Making the records
role single-site is what moves the work, and a scenario does not re-tune
its roster — those two rules together dropped five sites' headcount
entries on the floor, leaving three people holding the whole network's
retrieval. The run reported $13.9M, 151 hires and a 28-FTE peak gap, all
of it transition cost that does not exist. A hub re-bases people; it does
not make them redundant.

**A reported volume was wrong by four times.** Reactive digitization was
reported at about 79,000 charts a plan year and corrected to about 21,000.

**The digitization archive was recalibrated against the history.** The
first defaults implied a falling non-digital share; the generated history
shows a flat one. Where an assumption and the history disagree, the
assumption is wrong. Recalibrated, the project removes about 33 net hours
a year for $48,000 — roughly a 39-year payback.

**Cost became all-in.** Scenario comparisons had been quoting labour cost
only, which credited every automation scenario with its saving and none of
its cost. Labour, machine and all-in now sit side by side, with all-in as
the headline.

**A finding about integer hiring.** The fully-realized automation run
needs 0.94 fewer FTE than the half-realized one and yet hires *more*
people — seven in three classes against six in two. A class is sized to
the current gap and rounded up, so a smaller gap can buy a smaller class
that triggers again later. Total hires do not follow required hours in a
straight line. Investigated and confirmed as correct behaviour rather than
a defect.

## Session 6 — the Excel workbook (3–6 November, 8 commits)

Ten tabs a leader can open without Python. One design decision governed
all of it: the model writes every **measure** as a value, so it ties to
the model to the cent, and Excel computes every **total** and
**difference** as a live formula. Nothing is calculated twice in two
languages. The Summary tab holds no numbers of its own at all — every
figure on it is a reference into the scenario tab, so the two cannot
drift.

There is deliberately **no run timestamp** in the file, because a
wall-clock date would make two runs of a fixed-seed model produce
different workbooks.

**Testing formulas without Excel.** Excel is not available to the test
suite, and the library that writes the workbook never computes a formula.
So the tests parse the small formula language the exporter emits and
evaluate it against the values the workbook itself holds. Stated as a
limit: that proves the references point where they should and the
arithmetic is what it claims. It is not Excel.

### Six defects found, and how each was proven

Four were found by the tie-out suite, one by reading the output, one by
the architect reading the finished tab.

**A formula with no number format.** The hours-gap column wrote live
formulas with no format at all, so Excel would have shown a shortfall as
`-12.3456789` instead of `(12.3)`. Caught by the format test.

**Blank cells that could not be told apart from a failure.** Four inputs
on the Assumptions tab wrote empty cells — and the four did not mean the
same thing: one meant *choose this from the data*, one *no cap at all*,
one *every eligible type*. A blank cannot be distinguished from an
exporter that failed to write. They now say which kind of empty they are.
Caught by the cell-by-cell tie-out, on row 39 of 986 — which is the
argument for checking every row rather than a sample.

**The same blank again, elsewhere.** The automation ranking's medium
column went blank for work types whose handle time does not vary by
medium. Now `n/a`, with the tab saying what that means.

**A double count on the coverage log.** The plan-year total summed every
hours column down the log, reading **839.9 uncovered hours against a true
420.0** — exactly double, because the same uncovered hour is logged twice:
once on the overtime event as what was still short afterwards, and again
on the event that accepted the gap. That is the worst shape a wrong number
takes — plausible, quotable, and on the tab a leader reads to find out how
much work did not get done. The hours now total in a by-event-type block
where each hour is counted once, and that block is where two exact
cross-tab ties live.

**A header naming the wrong run.** The shared writer labelled every tab's
source line "Baseline run", which is right for four tabs and wrong for the
automation ranking, which is priced at the full-programme run. Found by
dumping the tab to plain text to show the architect — every figure on it
was right and every tie-out passed. The label was wrong.

**A saving priced as if nothing were automated.** The ranking's total
multiplied each per-item saving by every item of that work type that
*arrives*, describing a machine doing 100% of the work. It does about 8%.
The figure read −$1,459,130 against an honest −$87,770, seventeen times
too large — and not wrong in a safe direction, because it scaled up the
seventeen losing rows as well as the two winning ones. Found by the
architect reading the tab.

Worth recording that the tie-out suite did **not** catch that one. Every
cell matched the builder's own figures and every formula evaluated
correctly; the one number nobody had an independent route to was the one
that was wrong. A tie-out proves two things agree. It cannot notice that
both are answering a question nobody asked.

**All six were then proven by reverting the fix and confirming the test
fails.** Three had failed naturally against the broken code when they were
first run. The other three had their tests written *after* the fix, so no
test had ever actually failed against the bug it was written for — so
those three defects were reintroduced exactly as they had been, the suite
was run against them, all four covering tests failed, and the code was
restored from the commit.

### What the workbook's tie-out checks

Three layers, because a detail tab can be wrong in two independent places
and testing one proves nothing:

- **the writer** can put the right data in the wrong cells, so every
  written cell is compared to the builder's own frame, every row;
- **the builder** can hand over data that is not the model's answer, so
  its output is compared to the model's own tables, recomputed in the test
  — never by calling the builder twice, which would pass on any bug the
  builder has;
- **the totals** can be right per cell and still not add up, so every sum
  formula is parsed, evaluated and matched to the model's total.

And where two tabs reach the same figure by different routes, both are
checked to agree with neither allowed to be the reference.

### One reconciliation that does not hold, reported rather than forced

The automation ranking's total does not reconcile to the full-programme
run's labour saving less its machine cost: −$87,770 against −$141,312, a
gap of $53,543. The machine side of the table ties to the run's own
invoice to the cent, and the human side ties to the required hours that
run frees priced at loaded rates, to three cents. So the whole gap sits
between *required hours* and *payroll* — which is this project's recurring
finding with a number on it: a freed hour is only money where it cancels
an overtime hour or a hiring class, and on a held roster most do not.

The test pins that identity rather than asserting a reconciliation that is
not true, and the tab says so in words, because a reader comparing the two
tabs would otherwise conclude one of them is broken.

---

## What the build produced

655 tests across 21 files. The validation checks named in the
specification are tests, and a session's work was not finished until they
passed.

The habits that mattered most, in order of how much they caught:

1. **Hand-build the reference case first.** Two Excel workbooks worked out
   before the code existed, committed here, and read by the tests — which
   fail if the test constants and the workbook cells ever drift apart.
2. **Check every row, not a sample.** The offset and the single bad row
   are invisible in a sample, because every value you look at is some real
   value from the table.
3. **Make tests fail on purpose.** A test that has never failed has not
   been checked. Every defect in this log was proven by reverting its fix.
4. **Assert that the thing you measure actually varies.** A test that
   would pass whether or not it does is a test that will eventually
   mislead.
5. **Print the output and read it.** Two defects here were found that way
   and by no test — including one where every figure was correct and only
   the label was wrong.
6. **Write down what was withdrawn.** Three published findings in this
   project turned out to be defects. A build log that only lists successes
   is not a record of anything.
