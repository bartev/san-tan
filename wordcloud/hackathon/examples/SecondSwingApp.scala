package com.identified.bvoneoffs.hackathon.examples

import swing._
import swing.event.ButtonClicked

object SecondSwingApp extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "2nd swing app"
    val button = new Button {text = "Click me"}
    val button2 = new Button("click me too!")

    val label = new Label("No button clicks registered")

    contents = new BoxPanel(Orientation.Vertical) {
      contents += button
      contents += button2
      contents += label
      // Border: top, left, bottom, right
      border = Swing.EmptyBorder(100, 30, 0, 300)
    }
  }
}
