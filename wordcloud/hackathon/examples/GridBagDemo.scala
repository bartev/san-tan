package com.identified.bvoneoffs.hackathon.examples

import swing.{MainFrame, Button, GridBagPanel, SimpleSwingApplication}
import swing.GridBagPanel.{Anchor, Fill}
import java.awt.Insets

object GridBagDemo extends SimpleSwingApplication{
  lazy val ui = new GridBagPanel {
    val c = new Constraints
    val shouldFill = true
    if (shouldFill) {
      c.fill = Fill.Horizontal
    }

    val button1 = new Button("Button 1 (0, 0)")
    c.weightx=0.5
    c.fill = Fill.Horizontal
    c.gridx = 0
    c.gridy = 0
    layout(button1) = c

    val button2 = new Button("Button 2 (1, 0)")
    c.gridx = 1
    c.gridy = 0
    layout(button2) = c

    val button3 = new Button("Button 3 (2, 0)")
    c.gridx = 2
    c.gridy = 0
    layout(button3) = c

    val button4 = new Button("Button 4 (5, 4)")
    c.gridx = 5
    c.gridy = 4
    layout(button4) = c

    val button5 = new Button("Long named button 5 (0, 1)")
    c.ipady = 40; // make this componenet tall
    c.weightx = 0
    c.gridwidth = 4
    c.gridx = 0
    c.gridy = 1
    layout(button5) = c

    val button6 = new Button("5 (1, 4)")
    c.fill = Fill.Horizontal
    c.ipady = 0                         // reseet to default
    c.weighty = 1                       // request any extra vertical space
    c.anchor = Anchor.PageEnd
    c.insets = new Insets(10, 0, 0, 0)  // top padding
    c.gridx = 1                         // aligned with button 2
    c.gridwidth = 2                     // 2 columns wide
    c.gridy = 4                         // 5th row
    layout(button6) = c


  }

  def top = new MainFrame(){
    title = "GridBag Demo"
    contents = ui
  }
}
