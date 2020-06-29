# Imports
using EpistemicNetworkAnalysis
using DataFrames
using CSV
using Plots
using GLM
using Statistics
using MixedModels
using HypothesisTests

# cond UserName
# group by cond
# means on cond
# exclude second half
# drop coll and client

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
    plottitle::String, filename::String, width::Int, height::Int, xlabel::String, ylabel::String, showunits::Bool)

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
means = MeansRotation(:Condition, "FirstGame", "SecondGame")
univariate = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + CONFIDENCE_Change), nothing,
    LinearModel, 2, @formula(y ~ 1 + Condition), Dict(:Condition => DummyCoding(base="FirstGame"))
)

nested = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 2, @formula(y ~ 1 + CONFIDENCE_Change + CondGroupName + (1 + CONFIDENCE_Change|CondGroupName)), Dict(:CondGroupName => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Condition), Dict(:Condition => DummyCoding(base="FirstGame"))
)

# Artists
black = DefaultArtist()
# simple = MeansArtist(:NewC_Change, "No.Change", "Pos.Change")
simple = MeansArtist(:Condition, "FirstGame", "SecondGame")

# Requests
please(means, simple, ["FirstGame"], "Means", "means-simple-first", 1, 1, "First/Second Game", "SVD", true)
please(means, simple, ["SecondGame"], "Means", "means-simple-second", 1, 1, "First/Second Game", "SVD", true)
please(means, simple, ["FirstGame", "SecondGame"], "Means", "means-simple", 1, 1, "First/Second Game", "SVD", false)
please(univariate, simple, ["FirstGame", "SecondGame"], "Univariate", "univariate-simple", 1, 1, "Confidence Change", "First/Second Game", false)
please(nested, simple, ["FirstGame", "SecondGame"], "Nested", "nested-simple", 1, 1, "Confidence Change", "First/Second Game", false)