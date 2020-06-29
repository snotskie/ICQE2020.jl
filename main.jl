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
Genders = Dict(
    "steven z" => "Male",
    "akash v" => "Male",
    "alexander b" => "Male",
    "brandon l" => "Male",
    "christian x" => "Male",
    "jordan l" => "Male",
    "arden f" => "Male",
    "margaret n" => "Female",
    "connor f" => "Male",
    "jimmy i" => "Male",
    "devin c" => "Male",
    "tiffany x" => "Female",
    "amelia n" => "Female",
    "luis t" => "Male",
    "amalia x" => "Female",
    "robert z" => "Male",
    "joseph k" => "Male",
    "peter h" => "Male",
    "carl b" => "Male",
    "mitchell h" => "Male",
    "peter s" => "Male",
    "joseph h" => "Male",
    "cameron k" => "Male",
    "fletcher l" => "Male",
    "amirah u" => "Female",
    "kevin g" => "Male",
    "brent p" => "Male",
    "kiana k" => "Female",
    "madeline g" => "Female",
    "justin y" => "Male",
    "ruzhen e" => "Male",
    "brandon f" => "Male",
    "jackson p" => "Male",
    "shane t" => "Male",
    "samuel o" => "Male",
    "casey f" => "Male",
    "keegan q" => "Male",
    "nicholas l" => "Male",
    "cameron i" => "Male",
    "cormick u" => "Male",
    "daniel t" => "Male",
    "christina b" => "Female",
    "derek v" => "Male",
    "nicholas n" => "Male",
    "abigail z" => "Female",
    "caitlyn y" => "Female",
    "nathan d" => "Male",
    "luke u" => "Male"
)
RSData[!, :Gender] = map(RSRow -> Genders[RSRow[:UserName]], eachrow(RSData))
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
    plottitle::String, filename::String, width::Int, height::Int, xlabel::String, ylabel::String)

    ## Run model
    ena = ENAModel(
        RSData, codes, conversations, units,
        rotateBy=rotation, windowSize=5,
        subsetFilter=RSRow -> RSRow[:Condition] in ["FirstGame", "SecondGame"] && RSRow[:GameHalf] in ["First"]
    )

    display(ena)
   
    ## Plotting
    p = plot(
        ena,
        artist=artist,
        yaxisname=ylabel,
        xaxisname=xlabel,
        displayFilter=RSRow -> RSRow[:Gender] in displayset,
        showconfidence=false
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
    m1 = fit(
        MixedModel,
        @formula(CONFIDENCE_Change ~ 1 + pos_x + Gender + CondGroupName + pos_x&CondGroupName + Gender&CondGroupName + (1 + pos_x + Gender|CondGroupName)),
        ena.refitUnitModel, contrasts=Dict(:CondGroupName => EffectsCoding(), :Gender => EffectsCoding(), :Condition => EffectsCoding())
    )

    display(m1)
    # println(var(predict(m1)) / var(residuals(m1) + predict(m1)))
end

# Rotations
interceptOnly = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 1, @formula(y ~ 1 + 0), nothing,
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

interceptNested = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 1, @formula(y ~ 1 + CondGroupName + (1|CondGroupName)), Dict(:CondGroupName => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

interceptGendered = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 1, @formula(y ~ 1 + Gender), Dict(:Gender => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

univariate = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + CONFIDENCE_Change), nothing,
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

moderated = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 2, @formula(y ~ 1 + CONFIDENCE_Change + Gender + Gender&CONFIDENCE_Change), Dict(:Gender => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

nested = EpistemicNetworkAnalysis.Formula2Rotation(
    MixedModel, 2, @formula(y ~ 1 + CONFIDENCE_Change + CondGroupName + (1 + CONFIDENCE_Change|CondGroupName)), Dict(:CondGroupName => EffectsCoding()),
    LinearModel, 2, @formula(y ~ 1 + Gender), nothing
)

# Artists
black = DefaultArtist()
means = MeansArtist(:Gender, "Male", "Female")

# Requests
please(interceptOnly, black, ["Male", "Female"], "Intercept Only", "intercept-black", 1, 1, "Mean", "Female/Male")
# please(interceptOnly, means, ["Female"], "Intercept Only", "intercept-means-female", 1, 1, "Mean", "Female/Male")
# please(interceptOnly, means, ["Male"], "Intercept Only", "intercept-means-male", 1, 1, "Mean", "Female/Male")
# please(interceptOnly, means, ["Male", "Female"], "Intercept Only", "intercept-means", 1, 1, "Mean", "Female/Male")
please(interceptNested, means, ["Male", "Female"], "Intercept Nested", "intnested-means", 1, 1, "Mean", "Female/Male")
please(interceptGendered, means, ["Male", "Female"], "Intercept Gendered", "gendered-means", 1, 1, "Mean", "Female/Male")
please(univariate, means, ["Male", "Female"], "Univariate", "univariate-means", 1, 1, "Confidence Change", "Female/Male")
# please(moderated, means, ["Male", "Female"], "Moderated", "moderated-means", 1, 1, "Confidence Change", "Female/Male")
please(nested, means, ["Male", "Female"], "Nested", "nested-means", 1, 1, "Confidence Change", "Female/Male")

error("stop here")

# Data preprocessing
TADMUS_Data = DataFrame(CSV.File("../TADMUS/data/TADMUSCoded.csv"))

# ENA model config
conversations = [:Team, :Scenario]
units = [:COND, :Team, :MacroRole, :Speaker, :Scenario]
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

# Helper
function please(rotation::ENARotation, artist::ENAArtist, displayset::Array{String,1},
    plottitle::String, filename::String, width::Int, height::Int, xlabel::String, ylabel::String)

    ## Run model
    ena = ENAModel(
        TADMUS_Data, codes, conversations, units,
        rotateBy=rotation, windowSize=5,
        subsetFilter=TADMUS_Row -> TADMUS_Row[:Speaker] in ["TAO", "CO"]
    )

    display(ena)
   
    ## Plotting
    p = plot(
        ena,
        artist=artist,
        yaxisname=ylabel,
        xaxisname=xlabel,
        displayFilter=TADMUS_Row -> TADMUS_Row[:COND] in displayset
    )

    xlims!(p, -width, width)
    ylims!(p, -height, height)
    xticks!(p, [-width, 0, width])
    yticks!(p, [-height, 0, height])
    title!(p, plottitle)
    savefig(p, "$(filename).svg")
    savefig(p, "$(filename).png")
    display(p)
end

interceptOnly = EpistemicNetworkAnalysis.Formula2Rotation(
    LinearModel, 1, @formula(y ~ 1 + 0), nothing,
    LinearModel, 2, @formula(y ~ 1 + COND), nothing
)

plainMeans = EpistemicNetworkAnalysis.FormulaRotation(
    LinearModel, 2, @formula(y ~ 1 + Speaker), nothing
)

moderatedMeans = EpistemicNetworkAnalysis.FormulaRotation(
    LinearModel, 2, @formula(y ~ 1 + Speaker + COND + Scenario), Dict(:COND => EffectsCoding(), :Scenario => EffectsCoding())
)

black = DefaultArtist()
# means = MeansArtist(:COND, "Control", "Experimental")
means = MeansArtist(:Speaker, "TAO", "CO")
# please(interceptOnly, black, ["Control", "Experimental"], "Intercept Only", "intercept-black", 1, 1, "Mean", "Control/Treatment")
# please(interceptOnly, means, ["Control"], "Intercept Only", "intercept-control", 1, 1, "Mean", "Control/Treatment")
# please(interceptOnly, means, ["Experimental"], "Intercept Only", "intercept-treatment", 1, 1, "Mean", "Control/Treatment")
# please(interceptOnly, means, ["Control", "Experimental"], "Intercept Only", "intercept-subtract", 1, 1, "Mean", "Control/Treatment")
please(plainMeans, means, ["Control", "Experimental"], "Plain Means", "plain-subtract", 1, 1, "Control/Treatment", "SVD")
please(moderatedMeans, means, ["Control", "Experimental"], "Moderated Means", "plain-subtract", 1, 1, "Control/Treatment", "SVD")


# # Control_Data = filter(TADMUS_Row -> TADMUS_Row[:COND] == "Control", TADMUS_Data)
# # Treatment_Data = filter(TADMUS_Row -> TADMUS_Row[:COND] == "Experimental", TADMUS_Data)
# # TADMUS_Data = vcat(Control_Data, Treatment_Data)
# # TADMUS_Data = vcat(Treatment_Data, Control_Data)
# # NewTeamNames = Dict(old => "Team $i" for (i, old) in enumerate(unique(TADMUS_Data[!, :Team])))
# # TADMUS_Data[!, :Team] = map(x -> NewTeamNames[x], TADMUS_Data[!, :Team])

# TADMUS_Data[!, :DummyMacroRole] = map(eachrow(TADMUS_Data)) do TADMUS_Row
#     if TADMUS_Row[:MacroRole] == "Command"
#         return 1
#     else
#         return 0
#     end
# end

# TADMUS_Data[!, :DummyCOND] = map(eachrow(TADMUS_Data)) do TADMUS_Row
#     if TADMUS_Row[:COND] == "Experimental"
#         return 1
#     else
#         return 0
#     end
# end

# # ENA model config
# conversations = [:Team, :Scenario]
# units = [:COND, :Team, :MacroRole, :Speaker]
# codes = [
#     :SeekingInformation,
#     :DetectIdentify,
#     :TrackBehavior,
#     :StatusUpdate,
#     :AssessmentPrioritization,
#     :DefensiveOrders,
#     :DeterrentOrders,
#     :Recommendation
# ]

# # Inspecting the data
# println(names(TADMUS_Data))
# display(first(TADMUS_Data[!, unique([conversations..., units..., codes...])], 6))
# display(unique(TADMUS_Data[!, :COND]))
# display(unique(TADMUS_Data[!, :Team]))
# display(unique(TADMUS_Data[!, :Scenario]))
# display(unique(TADMUS_Data[!, :MacroRole]))
# display(unique(TADMUS_Data[!, :Speaker]))
# display(unique(TADMUS_Data[!, units]))
# display(unique(TADMUS_Data[!, conversations]))

# # Running ENA Models
# ## MR1
# myRotation1 = MeansRotation(:MacroRole, "Support", "Command")
# myENA1 = ENAModel(TADMUS_Data, codes, conversations, units,
#     rotateBy=myRotation1,
#     windowSize=5,
#     subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
# )

# ## F2
# myRotation2 = EpistemicNetworkAnalysis.Formula2Rotation(
#     LinearModel, 2, @formula(y ~ 1 + DummyMacroRole), nothing,
#     LinearModel, 2, @formula(y ~ 1 + DummyCOND), nothing
# )

# myENA2 = ENAModel(TADMUS_Data, codes, conversations, units,
#     rotateBy=myRotation2,
#     windowSize=5,
#     subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
# )

# ## HLM
# myRotation3 = EpistemicNetworkAnalysis.Formula2Rotation(
#     MixedModel, 2, @formula(y ~ 1 + DummyMacroRole + (1 + DummyMacroRole|Team)), Dict(:Team => EffectsCoding()),
#                 # NOTE x-axis should use a continous or 0/1 dummy with dummycoding, to be interpretable correctly. the moderating categorical vars should use effectscoding, so that we are interpreting the slope when in the grand mean, not when in an arbitrary reference group
#     # LinearModel, 2, @formula(y ~ 1 + DummyMacroRole + AVG_LEADTOT), nothing,
#     MixedModel, 2, @formula(y ~ 1 + DummyCOND + (1 + DummyCOND|Team)), Dict(:Team => EffectsCoding())
# )

# myENA3 = ENAModel(TADMUS_Data, codes, conversations, units,
#     rotateBy=myRotation3,
#     windowSize=5,
#     subsetFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command", "Support"]
# )

# # Displaying descriptive results
# display(myENA1)
# display(myENA2)
# display(myENA3)

# # Plotting
# myArtist = EpistemicNetworkAnalysis.TVRemoteArtist(
#     :DummyMacroRole, 0, 1,
#     :DummyCOND, 0, 1
# )

# ## MR1
# p1 = plot(myENA1,
#     artist=myArtist,
#     yaxisname="SVD1",
#     xaxisname="Support/Command",
#     displayFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Support"]
# )

# k = 1
# xlims!(p1, -k, k)
# ylims!(p1, -k, k)
# title!(p1, "Means Rotation (Support)")
# savefig(p1, "mr1-support.svg")
# savefig(p1, "mr1-support.png")
# display(p1)

# p2 = plot(myENA1,
#     artist=myArtist,
#     yaxisname="SVD1",
#     xaxisname="Support/Command",
#     displayFilter=TADMUS_Row -> TADMUS_Row[:MacroRole] in ["Command"]
# )

# xlims!(p2, -k, k)
# ylims!(p2, -k, k)
# title!(p2, "Means Rotation (Command)")
# savefig(p2, "mr1-command.svg")
# savefig(p2, "mr1-command.png")
# display(p2)

# p3 = plot(myENA1,
#     artist=myArtist,
#     yaxisname="SVD",
#     xaxisname="Support/Command",
#     showunits=false
# )

# xlims!(p3, -k, k)
# ylims!(p3, -k, k)
# title!(p3, "Means Rotation (Subtraction)")
# savefig(p3, "mr1-subtract.svg")
# savefig(p3, "mr1-subtract.png")
# display(p3)

# ## F2
# k = 1
# p4 = plot(myENA2,
#     artist=myArtist,
#     yaxisname="Control/Treatment",
#     xaxisname="Support/Command",
#     showunits=false
# )

# xlims!(p4, -k, k)
# ylims!(p4, -k, k)
# title!(p4, "Univariate")
# savefig(p4, "univariate.svg")
# savefig(p4, "univariate.png")
# display(p4)

# ## HLM
# k = 1
# p5 = plot(myENA3,
#     artist=myArtist,
#     yaxisname="Control/Treatment",
#     xaxisname="Support/Command",
#     showunits=false
# )

# xlims!(p5, -k, k)
# ylims!(p5, -k, k)
# title!(p5, "Hierarchical")
# savefig(p5, "hierarchical.svg")
# savefig(p5, "hierarchical.png")
# display(p5)

# # Hypothesis tests
# # # m1 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA1.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))
# # # m2 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA2.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))
# # # m3 = fit(MixedModel, @formula(DummyMacroRole ~ 1 + pos_x + Team + pos_x&Team + (1 + pos_x|Team)), myENA3.refitUnitModel, contrasts=Dict(:Team => EffectsCoding()))

# # m1 = fit(LinearModel, @formula(DummyMacroRole ~ 1 + pos_x + AVG_LEADTOT), myENA1.refitUnitModel)
# # m2 = fit(LinearModel, @formula(DummyMacroRole ~ 1 + pos_x + AVG_LEADTOT), myENA2.refitUnitModel)
# # m3 = fit(LinearModel, @formula(DummyMacroRole ~ 1 + pos_x + AVG_LEADTOT), myENA3.refitUnitModel)

# # display(m1)
# # display(m2)
# # display(m3)

# # println(var(predict(m1)) / var(residuals(m1) + predict(m1))) # 0.6119
# # println(var(predict(m2)) / var(residuals(m2) + predict(m2))) # 0.6119 (same x-axis as above)
# # println(var(predict(m3)) / var(residuals(m3) + predict(m3))) # 0.6134

# # # Idea
# # RSData = ena_dataset("RS.data")
# # Genders = Dict(
# #     "steven z" => "Male",
# #     "akash v" => "Male",
# #     "alexander b" => "Male",
# #     "brandon l" => "Male",
# #     "christian x" => "Male",
# #     "jordan l" => "Male",
# #     "arden f" => "Male",
# #     "margaret n" => "Female",
# #     "connor f" => "Male",
# #     "jimmy i" => "Male",
# #     "devin c" => "Male",
# #     "tiffany x" => "Female",
# #     "amelia n" => "Female",
# #     "luis t" => "Male",
# #     "amalia x" => "Female",
# #     "robert z" => "Male",
# #     "joseph k" => "Male",
# #     "peter h" => "Male",
# #     "carl b" => "Male",
# #     "mitchell h" => "Male",
# #     "peter s" => "Male",
# #     "joseph h" => "Male",
# #     "cameron k" => "Male",
# #     "fletcher l" => "Male",
# #     "amirah u" => "Female",
# #     "kevin g" => "Male",
# #     "brent p" => "Male",
# #     "kiana k" => "Female",
# #     "madeline g" => "Female",
# #     "justin y" => "Male",
# #     "ruzhen e" => "Male",
# #     "brandon f" => "Male",
# #     "jackson p" => "Male",
# #     "shane t" => "Male",
# #     "samuel o" => "Male",
# #     "casey f" => "Male",
# #     "keegan q" => "Male",
# #     "nicholas l" => "Male",
# #     "cameron i" => "Male",
# #     "cormick u" => "Male",
# #     "daniel t" => "Male",
# #     "christina b" => "Female",
# #     "derek v" => "Male",
# #     "nicholas n" => "Male",
# #     "abigail z" => "Female",
# #     "caitlyn y" => "Female",
# #     "nathan d" => "Male",
# #     "luke u" => "Male"
# # )
# # RSData[!, :Gender] = map(RSRow -> Genders[RSRow[:UserName]], eachrow(RSData))
# # conversations = [:Condition, :GameHalf, :GroupName]
# # units = [:Condition, :GameHalf, :UserName]
# # codes = [
# #     :Data,
# #     :Technical_Constraints,
# #     :Performance_Parameters,
# #     :Client_and_Consultant_Requests,
# #     :Design_Reasoning,
# #     :Collaboration
# # ]

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     LinearModel, 2, @formula(y ~ 1 + Condition + Gender + Condition&Gender), Dict(:Condition => EffectsCoding(), :Gender => EffectsCoding()),
# # #     LinearModel, 2, @formula(y ~ 1 + GameHalf), Dict(:GameHalf => EffectsCoding())
# # # )

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     LinearModel, 2, @formula(y ~ 1 + Gender), nothing,
# # #     LinearModel, 2, @formula(y ~ 1 + GameHalf), nothing
# # # )

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     LinearModel, 2, @formula(y ~ 1 + CONFIDENCE_Pre), nothing,
# # #     MixedModel, 2, @formula(y ~ 1 + CONFIDENCE_Post + (1 + CONFIDENCE_Post|GroupName)), Dict(:GroupName => EffectsCoding())
# # # )

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     MixedModel, 2, @formula(y ~ 1 + CONFIDENCE_Change + (1 + CONFIDENCE_Change|GroupName)), Dict(:GroupName => EffectsCoding()),
# # #     LinearModel, 2, @formula(y ~ 1 + Gender + GroupName), Dict(:GroupName => EffectsCoding())
# # # )

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     LinearModel, 2, @formula(y ~ 1 + GameDay + GroupName + Condition&GameDay), Dict(:GroupName => EffectsCoding()),
# # #     LinearModel, 2, @formula(y ~ 1 + Gender + GroupName), Dict(:GroupName => EffectsCoding())
# # #     # LinearModel, 2, @formula(y ~ 1 + Condition + GroupName), Dict(:GroupName => EffectsCoding())
# # # )

# # # myRotation4 = EpistemicNetworkAnalysis.Formula2Rotation(
# # #     MixedModel, 2, @formula(y ~ 1 + Gender + (1 + Gender|GroupName)), Dict(:GroupName => EffectsCoding()),
# # #     LinearModel, 2, @formula(y ~ 1 + GameHalf), nothing
# # # )

# # myRotation4 = EpistemicNetworkAnalysis.FormulaRotation(
# #     # LinearModel, 2, @formula(y ~ 1 + Gender), nothing
# #     MixedModel, 2, @formula(y ~ 1 + Gender + (1 + Gender|GroupName)), Dict(:GroupName => EffectsCoding())
# # )

# # myENA4 = ENAModel(RSData, codes, conversations, units, rotateBy=myRotation4)
# # display(myENA4)
# # # myArtist = EpistemicNetworkAnalysis.TVRemoteArtist(
# # #     :Condition, "FirstGame", "SecondGame",
# # #     :GameHalf, "First", "Second"
# # # )

# # myArtist = MeansArtist(:Gender, "Male", "Female")
# # # myArtist = MeansArtist(:Condition, "FirstGame", "SecondGame")

# # # myArtist = DefaultArtist()

# # p5 = plot(myENA4, artist=myArtist, yaxisname="SVD", xaxisname="MR1")
# # # title!(p5, "Univariate")
# # # title!(p5, "Hierarchical")
# # title!(p5, "Moderated for GroupName")
# # xlims!(p5, -1, 1)
# # ylims!(p5, -1, 1)
# # savefig(p5, "gender.svg")
# # # savefig(p5, "gender2.svg")
# # display(p5)

# # # # m4 = fit(MixedModel, @formula(GameHalf ~ 1 + pos_x + GroupName + pos_x&GroupName + (1 + pos_x|GroupName)), myENA4.refitUnitModel, contrasts=Dict(:GroupName => EffectsCoding()))
# # # # display(m4)
# # # # println(var(predict(m4)) / var(residuals(m4) + predict(m4))) # 0.6119