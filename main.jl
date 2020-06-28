using EpistemicNetworkAnalysis
using DataFrames
using CSV
using Plots
using GLM
using Statistics
using MixedModels
using HypothesisTests

# Data preprocessing
TADMUS_Data = DataFrame(CSV.File("../TADMUS/data/TADMUSCoded.csv"))

# Control_Data = filter(TADMUS_Row -> TADMUS_Row[:COND] == "Control", TADMUS_Data)
# Treatment_Data = filter(TADMUS_Row -> TADMUS_Row[:COND] == "Experimental", TADMUS_Data)
# TADMUS_Data = vcat(Control_Data, Treatment_Data)
# TADMUS_Data = vcat(Treatment_Data, Control_Data)
# NewTeamNames = Dict(old => "Team $i" for (i, old) in enumerate(unique(TADMUS_Data[!, :Team])))
# TADMUS_Data[!, :Team] = map(x -> NewTeamNames[x], TADMUS_Data[!, :Team])

TADMUS_Data[!, :DummyMacroRole] = map(eachrow(TADMUS_Data)) do TADMUS_Row
    if TADMUS_Row[:MacroRole] == "Command"
        return 1
    else
        return 0
    end
end

TADMUS_Data[!, :DummyCOND] = map(eachrow(TADMUS_Data)) do TADMUS_Row
    if TADMUS_Row[:COND] == "Experimental"
        return 1
    else
        return 0
    end
end

# ENA model config
conversations = [:Team, :Scenario]
units = [:COND, :Team, :MacroRole, :Speaker]
codes = [
    :SeekingInformation,
    :DetectIdentify,
    :TrackBehavior,
    :StatusUpdate,
    :AssessmentPrioritization,
    :DefensiveOrders,
    :DeterrentOrders,
    :Recommendation
]

# Inspecting the data
println(names(TADMUS_Data))
display(first(TADMUS_Data[!, unique([conversations..., units..., codes...])], 6))
display(unique(TADMUS_Data[!, :COND]))
display(unique(TADMUS_Data[!, :Team]))
display(unique(TADMUS_Data[!, :Scenario]))
display(unique(TADMUS_Data[!, :MacroRole]))
display(unique(TADMUS_Data[!, :Speaker]))
display(unique(TADMUS_Data[!, units]))
display(unique(TADMUS_Data[!, conversations]))

# Running ENA Models
## MR1
myRotation1 = MeansRotation(:MacroRole, "Support", "Command")
myENA1 = ENAModel(TADMUS_Data, codes, conversations, units,
    rotateBy=myRotation1,
    windowSize=5,
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
)

## F2
myRotation2 = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + DummyMacroRole), nothing,
    LinearModel, 2, @formula(y ~ 1 + DummyCOND), nothing
)

myENA2 = ENAModel(TADMUS_Data, codes, conversations, units,
    rotateBy=myRotation2,
    windowSize=5,
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
)

## HLM
myRotation3 = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 2, @formula(y ~ 1 + DummyMacroRole + Team + DummyMacroRole&Team + (1 + DummyMacroRole|Team)), Dict(:Team => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + DummyCOND), nothing
)

myENA3 = ENAModel(TADMUS_Data, codes, conversations, units,
    rotateBy=myRotation3,
    windowSize=5,
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
)

# Displaying descriptive results
display(myENA1)
display(myENA2)
display(myENA3)

# Plotting
myArtist = EpistemicNetworkAnalysis.TVRemoteArtist(
    :DummyMacroRole, 0, 1,
    :DummyCOND, 0, 1
)

## MR1
p1 = plot(myENA1,
    artist=myArtist,
    yaxisname="SVD1",
    xaxisname="Support/Command",
    displayFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Support"]
)

k = 1
xlims!(p1, -k, k)
ylims!(p1, -k, k)
title!(p1, "Means Rotation (Support)")
savefig(p1, "mr1-support.svg")
savefig(p1, "mr1-support.png")
display(p1)

p2 = plot(myENA1,
    artist=myArtist,
    yaxisname="SVD1",
    xaxisname="Support/Command",
    displayFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command"]
)

xlims!(p2, -k, k)
ylims!(p2, -k, k)
title!(p2, "Means Rotation (Command)")
savefig(p2, "mr1-command.svg")
savefig(p2, "mr1-command.png")
display(p2)

p3 = plot(myENA1,
    artist=myArtist,
    yaxisname="SVD",
    xaxisname="Support/Command",
    showunits=false
)

xlims!(p3, -k, k)
ylims!(p3, -k, k)
title!(p3, "Means Rotation (Subtraction)")
savefig(p3, "mr1-subtract.svg")
savefig(p3, "mr1-subtract.png")
display(p3)

## F2
k = 1
p4 = plot(myENA2,
    artist=myArtist,
    yaxisname="Control/Treatment",
    xaxisname="Support/Command",
    showunits=false
)

xlims!(p4, -k, k)
ylims!(p4, -k, k)
title!(p4, "Univariate")
savefig(p4, "univariate.svg")
savefig(p4, "univariate.png")
display(p4)

## HLM
k = 1
p5 = plot(myENA3,
    artist=myArtist,
    yaxisname="Control/Treatment",
    xaxisname="Support/Command",
    showunits=false
)

xlims!(p5, -k, k)
ylims!(p5, -k, k)
title!(p5, "Hierarchical")
savefig(p5, "hierarchical.svg")
savefig(p5, "hierarchical.png")
display(p5)

# Hypothesis tests
m1 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA1.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))
m2 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA2.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))
m3 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA3.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))

display(m1)
display(m2)
display(m3)

println(var(predict(m1)) / var(residuals(m1) + predict(m1))) # 0.6119
println(var(predict(m2)) / var(residuals(m2) + predict(m2))) # 0.6119 (same x-axis as above)
println(var(predict(m3)) / var(residuals(m3) + predict(m3))) # 0.6134