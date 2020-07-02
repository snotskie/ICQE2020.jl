# Imports
using EpistemicNetworkAnalysis
using DataFrames
using CSV
using Plots
using GLM
using Statistics
using MixedModels
using HypothesisTests

# Data
RSData = ena_dataset("RS.data")

# Specification
conversations = [:Condition, :GameHalf, :GroupName]
units = [:Condition, :GameHalf, :UserName]
codes = [
    :Data,
    :Technical_Constraints,
    :Performance_Parameters,
    :Client_and_Consultant_Requests,
    :Design_Reasoning,
    :Collaboration
]

# Rotations
means = MeansRotation(:Condition, "FirstGame", "SecondGame", flipsvd=true)
means2 = MeansRotation(:GameHalf, "First", "Second")
interaction = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + Condition + GameHalf + Condition&GameHalf), Dict(:Condition => EffectsCoding(), :GameHalf => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + GameHalf), nothing #Dict(:CondGroupName => EffectsCoding())
)

# Artist
tv = EpistemicNetworkAnalysis.TVRemoteArtist(
    :Condition, "FirstGame", "SecondGame",
    :GameHalf, "First", "Second"
)

tv2 = EpistemicNetworkAnalysis.TVRemoteArtist(
    :GameHalf, "First", "Second",
    :Condition, "FirstGame", "SecondGame"
)

# Model 1
ena = ENAModel(
    RSData, codes, conversations, units,
    rotateBy=means, windowSize=5
)

display(ena)
p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="SVD",
    displayFilter=RSRow -> RSRow[:Condition] in ["FirstGame"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, First Game")
savefig(p, "first-game.svg")
savefig(p, "first-game.png")
run(`inkscape first-game.svg -E first-game.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="SVD",
    displayFilter=RSRow -> RSRow[:Condition] in ["SecondGame"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, Second Game")
savefig(p, "second-game.svg")
savefig(p, "second-game.png")
run(`inkscape second-game.svg -E second-game.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="SVD"
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, Subtraction")
savefig(p, "subtraction.svg")
savefig(p, "subtraction.png")
run(`inkscape subtraction.svg -E subtraction.eps --export-ignore-filters --export-ps-level=3`)
display(p)

# Model 2
ena = ENAModel(
    RSData, codes, conversations, units,
    rotateBy=interaction, windowSize=5,
)

display(ena)
p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="First/Second Half",
    displayFilter=RSRow -> RSRow[:Condition] in ["FirstGame"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Custom Rotation, First Game")
savefig(p, "interaction-first.svg")
savefig(p, "interaction-first.png")
run(`inkscape interaction-first.svg -E interaction-first.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="First/Second Half",
    displayFilter=RSRow -> RSRow[:Condition] in ["SecondGame"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Custom Rotation, Second Game")
savefig(p, "interaction-second.svg")
savefig(p, "interaction-second.png")
run(`inkscape interaction-second.svg -E interaction-second.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv,
    xaxisname="First/Second Game",
    yaxisname="First/Second Half"
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Custom Rotation, Subtraction")
savefig(p, "interaction.svg")
savefig(p, "interaction.png")
run(`inkscape interaction.svg -E interaction.eps --export-ignore-filters --export-ps-level=3`)
display(p)

# Model 3
ena = ENAModel(
    RSData, codes, conversations, units,
    rotateBy=means2, windowSize=5
)

display(ena)
p = plot(
    ena,
    artist=tv2,
    xaxisname="First/Second Half",
    yaxisname="SVD",
    displayFilter=RSRow -> RSRow[:GameHalf] in ["First"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, First Game")
savefig(p, "first-half.svg")
savefig(p, "first-half.png")
run(`inkscape first-half.svg -E first-half.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv2,
    xaxisname="First/Second Half",
    yaxisname="SVD",
    displayFilter=RSRow -> RSRow[:GameHalf] in ["Second"]
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, Second Game")
savefig(p, "second-half.svg")
savefig(p, "second-half.png")
run(`inkscape second-half.svg -E second-half.eps --export-ignore-filters --export-ps-level=3`)
display(p)

p = plot(
    ena,
    artist=tv2,
    xaxisname="First/Second Half",
    yaxisname="SVD"
)

xlims!(p, -1, +1)
ylims!(p, -1, +1)
title!(p, "Means Rotation, Subtraction")
savefig(p, "subtraction2.svg")
savefig(p, "subtraction2.png")
run(`inkscape subtraction2.svg -E subtraction2.eps --export-ignore-filters --export-ps-level=3`)
display(p)