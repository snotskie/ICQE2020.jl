# Imports
using EpistemicNetworkAnalysis
using DataFrames
using CSV
using Plots
using GLM
using Statistics
using MixedModels
using HypothesisTests

# Data preprocessing
RSData = ena_dataset("RS.data")
RSData[!, :CondGroupName] = string.(values.(eachrow(RSData[!, [:Condition, :GroupName]])))

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

# Helper
function please(rotation::ENARotation, artist::ENAArtist, displayset::Array{String,1},
    plottitle::String, filename::String, width::Real, height::Real, xlabel::String, ylabel::String, showunits::Bool)

    ## Run model
    ena = ENAModel(
        RSData, codes, conversations, units,
        rotateBy=rotation, windowSize=5,
        subsetFilter=RSRow -> RSRow[:Condition] in ["FirstGame", "SecondGame"] && RSRow[:GameHalf] in ["First"] && string(RSRow[:NewC_Change]) in ["Pos.Change", "No.Change"]
    )

    display(ena)
   
    ## Plotting
    p = plot(
        ena,
        artist=artist,
        yaxisname=ylabel,
        xaxisname=xlabel,
        displayFilter=RSRow -> RSRow[:Condition] in displayset,
        showunits=showunits
    )

    xlims!(p, -width, width)
    ylims!(p, -height, height)
    xticks!(p, [-width, 0, width])
    yticks!(p, [-height, 0, height])
    title!(p, plottitle)
    savefig(p, "$(filename).svg")
    savefig(p, "$(filename).png")
    display(p)

    ## Testing
    # m1 = fit(
    #     MixedModel,
    #     @formula(CONFIDENCE_Change ~ 1 + pos_x + Gender + CondGroupName + pos_x&CondGroupName + Gender&CondGroupName + (1 + pos_x + Gender|CondGroupName)),
    #     ena.refitUnitModel, contrasts=Dict(:CondGroupName => EffectsCoding(), :Gender => EffectsCoding(), :Condition => EffectsCoding())
    # )

    # display(m1)
    # println(var(predict(m1)) / var(residuals(m1) + predict(m1)))
end

# Rotations
means = MeansRotation(:NewC_Change, "No.Change", "Pos.Change")
binned = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + NewC_Change), Dict(:NewC_Change => DummyCoding(base="No.Change")),
    LinearModel, 2, @formula(y ~ 1 + Condition), Dict(:Condition => DummyCoding(base="FirstGame"))
)

univariate = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + CONFIDENCE_Change), nothing,
    LinearModel, 2, @formula(y ~ 1 + Condition), Dict(:Condition => DummyCoding(base="FirstGame"))
)

nested = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 2, @formula(y ~ 1 + CONFIDENCE_Change + CondGroupName + (1 + CONFIDENCE_Change|CondGroupName)), Dict(:CondGroupName => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Condition), Dict(:Condition => DummyCoding(base="FirstGame"))
)

# Artists
simple = MeansArtist(:Condition, "FirstGame", "SecondGame")
tv = EpistemicNetworkAnalysis.TVRemoteArtist(
    :Condition, "FirstGame", "SecondGame",
    :NewC_Change, "No.Change", "Pos.Change"
)

# Requests
please(means, tv, ["FirstGame"], "Means Rotation, First Game", "means-tv-first", 1, 1, "No/Pos Confidence Change", "SVD", true)
please(means, tv, ["SecondGame"], "Means Rotation, Second Game", "means-tv-second", 1, 1, "No/Pos Confidence Change", "SVD", true)
please(means, tv, ["FirstGame", "SecondGame"], "Means Rotation, Subtraction", "means-tv", 1, 1, "No/Pos Confidence Change", "SVD", false)
please(binned, tv, ["FirstGame", "SecondGame"], "Means Rotation, Both Axes", "binned-tv", 1, 1, "No/Pos Confidence Change", "First/Second Game", false)
please(univariate, tv, ["FirstGame", "SecondGame"], "Continuous Variable on X", "univariate-tv", 1, 1, "Confidence Change", "First/Second Game", false)
please(nested, tv, ["FirstGame", "SecondGame"], "Accounting for Nested Structure on X", "nested-tv", 1, 1, "Confidence Change", "First/Second Game", false)

please(means, tv, ["FirstGame"], "Means Rotation, First Game", "means-tv-first-detail", 0.4, 0.4, "No/Pos Confidence Change", "SVD", false)
please(means, tv, ["SecondGame"], "Means Rotation, Second Game", "means-tv-second-detail", 0.4, 0.4, "No/Pos Confidence Change", "SVD", false)
please(means, tv, ["FirstGame", "SecondGame"], "Means Rotation, Subtraction", "means-tv-detail", 0.4, 0.4, "No/Pos Confidence Change", "SVD", false)
please(binned, tv, ["FirstGame", "SecondGame"], "Means Rotation, Both Axes", "binned-tv-detail", 0.4, 0.4, "No/Pos Confidence Change", "First/Second Game", false)
please(univariate, tv, ["FirstGame", "SecondGame"], "Continuous Variable on X", "univariate-tv-detail", 0.4, 0.4, "Confidence Change", "First/Second Game", false)
please(nested, tv, ["FirstGame", "SecondGame"], "Accounting for Nested Structure on X", "nested-tv-detail", 0.4, 0.4, "Confidence Change", "First/Second Game", false)