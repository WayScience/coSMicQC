#let poster(
  size: "'36x24' or '48x36''",
  title: "Paper Title",
  authors: "Author Names (separated by commas)",
  departments: "Department Name",
  univ_logo: "Logo Path",
  footer_text: "Footer Text",
  footer_url: "Footer URL",
  footer_email_ids: "Email IDs (separated by commas)",
  footer_color: "Hex Color Code",
  keywords: (),
  num_columns: "4",
  univ_logo_scale: "210",
  univ_logo_column_size: "10",
  title_column_size: "25",
  title_font_size: "48",
  authors_font_size: "36",
  footer_url_font_size: "40",
  footer_text_font_size: "40",
  body
) = {
  set text(font: "Lato", size: 35pt)
  let sizes = size.split("x")
  let width = int(sizes.at(0)) * 1in
  let height = int(sizes.at(1)) * 1in
  univ_logo_scale = int(univ_logo_scale) * 1%
  title_font_size = int(title_font_size) * 1pt
  authors_font_size = int(authors_font_size) * 1pt
  num_columns = int(num_columns)
  univ_logo_column_size = int(univ_logo_column_size) * 1in
  title_column_size = int(title_column_size) * 1in
  footer_url_font_size = int(footer_url_font_size) * 1pt
  footer_text_font_size = int(footer_text_font_size) * 1pt

  set page(
    width: width,
    height: height,
    margin:
      (top: 1in, left: 1in, right: 1in, bottom: 2in),
    footer: [
      #set align(center)
      #set text(42pt)
      #block(
        fill: rgb(footer_color),
        width: 100%,
        inset: 20pt,
        radius: 10pt,
        [
          #text(font: "Lato", size: footer_url_font_size, footer_url)
          #h(1fr)
          #text(size: footer_text_font_size, smallcaps(footer_text))
          #h(1fr)
          #text(font: "Lato", size: footer_url_font_size, footer_email_ids)
        ]
      )
    ]
  )

  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)

  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  set heading(numbering: "I.A.1.")
  show heading: it => locate(loc => {
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(24pt, weight: 400)
    if it.level == 1 [
      #set text(style: "italic")
      #v(32pt, weak: true)
      #if it.numbering != none {
        numbering("i.", deepest)
        h(7pt, weak: true)
      }
      #it.body
    ] else if it.level == 2 [
      #v(10pt, weak: true)
      #set align(left)
      #set text({ 40pt }, weight: 600, font: "Merriweather", fill: rgb(31, 23, 112))
      #show: smallcaps
      #v(50pt, weak: true)
      #if it.numbering != none {
        numbering("I.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(30pt, weak: true)
      #line(length: 100%, stroke: rgb(200, 200, 200))
      #v(30pt, weak: true)
    ] else [
      #set text({ 32pt }, weight: 600, font: "Merriweather")
      #if it.level == 3 {
        numbering("☆ 1)", deepest)
        [ ]
      }
      ___#(it.body)___
      #v(40pt, weak: true)
    ]
  })

  align(left,
    grid(
      rows: (auto, auto),
      columns: (title_column_size, univ_logo_column_size),
      column-gutter: 10pt,
      row-gutter: 45pt,
      text(font: "Merriweather", weight: 1000, size: 58pt, title),
      grid.cell(
        image(univ_logo, width: univ_logo_scale),
        rowspan: 3
      ),
      text(size: 50pt, authors),
      text(size: 38pt, emph(departments)),
    )
  )

  v(60pt)

  show: columns.with(num_columns, gutter: 70pt)
  set par(leading: 10pt,
    justify: false,
    first-line-indent: 0em,
    linebreaks: "optimized"
  )

  body
}
