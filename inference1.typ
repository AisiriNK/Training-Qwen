#page( 
  paper: "a4",
  margin: (top: 1in, bottom: 1in, left: 1.25in, right: 1in), 
  header: none, 
  footer: none, 
  numbering: none 
)[
  #set text(font: "Times New Roman", size: 12pt, fill: black)
  #set align(center) 
  #v(1fr) 
  
  #text(size: 24pt, weight: "bold")[CHAPTER -- VII] 
  #v(0.5em) 
  #text(size: 24pt, weight: "bold")[METHODOLOGY AND IMPLEMENTATION] 
  #v(1fr) ] 
  #let starting_page = 8 
  #counter(page).update(starting_page+1) 
  #let department = "B.E/Dept of CSE/BNMIT" 
  #let academic_year = "2025-26" 
  #let project_title = "Sleep Apnea Detection" 
  #set page( 
    paper: "a4", 
    margin: (top: 1in, bottom: 1in, left: 1.25in, right: 1in), 
    numbering: "1", 
    header: context [ 
    #if counter(page).get().first() >= starting_page+2 [ 
      #set text(size: 12pt, font: "Times New Roman", weight: "bold") 
      #block( 
        width: 100%, 
        inset: (top: 8pt, bottom: 8pt), 
        )[ 
          #stack( 
            dir: ttb, 
            project_title, 
            v(0.3em), 
            spacing: 1.5pt, 
            line(length: 100%, stroke: 0.7pt + rgb(128, 0, 0)), 
            v(0.3em), 
            line(length: 100%, stroke: 3pt + rgb(128, 0, 0)) ) ] ]], 
            footer: context [
               #set text(size: 10pt, font: "Times New Roman") 
               #block( 
                width: 100%, 
                inset: (top: 8pt, bottom: 8pt), 
                )[ 
                  #stack( 
                    dir: ttb, 
                    spacing: 1.5pt, 
                    line(length: 100%, stroke: 3pt + rgb(128, 0, 0)),
                    v(0.3em),
                    line(length: 100%, stroke: 0.7pt + rgb(128, 0, 0))
                  )
                  #grid(
                    columns: (1fr, 1fr, 1fr),
                    align: (left, center, right),
                    department,
                    [Page #counter(page).display()],
                    academic_year
                  )
                ]
              ]
            )
                         