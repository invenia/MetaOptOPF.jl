using Documenter, GoCompetition

makedocs(;
    modules=[GoCompetition],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/research/GoCompetition.jl/blob/{commit}{path}#L{line}",
    sitename="GoCompetition.jl",
    authors="Invenia Technical Computing Corporation",
    assets=[
        "assets/invenia.css",
    ],
    strict=true,
    html_prettyurls=false,
    checkdocs=:none,
)
