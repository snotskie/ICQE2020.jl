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

# Running first ENA
myRotation = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 2, @formula(y ~ 1 + DummyMacroRole + Team + DummyMacroRole&Team + (1|Team) + (DummyMacroRole|Team)),
    LinearModel, 2, @formula(y ~ 1 + DummyCOND)
)

myENA = ENAModel(TADMUS_Data, codes, conversations, units,
    rotateBy=myRotation,
    windowSize=5,
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
)

display(myENA)

# Plotting
myArtist = EpistemicNetworkAnalysis.TVRemoteArtist(
    :DummyMacroRole, 0, 1,
    :DummyCOND, 0, 1
)

p1 = plot(myENA,
    artist=myArtist,
    showconfidence=true,
    yaxisname="Control/Treatment",
    xaxisname="Support/Command"
)

k = 1
xlims!(p1, -k, k)
ylims!(p1, -k, k)
title!(p1, "Controlled Model")
savefig(p1, "controlled.svg")
savefig(p1, "controlled.png")
display(p1)

# Running second ENA
myRotation = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + DummyMacroRole),
    LinearModel, 2, @formula(y ~ 1 + DummyCOND)
)

myENA = ENAModel(TADMUS_Data, codes, conversations, units,
    rotateBy=myRotation,
    windowSize=5,
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
)

display(myENA)

# Plotting
p2 = plot(myENA,
    artist=myArtist,
    showconfidence=true,
    yaxisname="Control/Treatment",
    xaxisname="Support/Command",
    subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command"]
)

k = 1
xlims!(p2, -k, k)
ylims!(p2, -k, k)
title!(p2, "Univariate Model")
savefig(p2, "uncontrolled.svg")
savefig(p2, "uncontrolled.png")
display(p2)

# TODO run each model on just one of the two groups