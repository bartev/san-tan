package com.identified.bvoneoffs.hackathon.examples

import swing._
import swing.event._

object ReactiveSwingApp extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "Reactive Swing App"
    val button = new Button("Click me")
    val button2 = new Button("click me too!")
    val button3 = new Button("reset")
    val label = new Label("No button clicks registered")

    contents = new BoxPanel(Orientation.Vertical) {
      contents += button
      contents += button2
      contents += button3
      contents += label
      // Border: top, left, bottom, right
      border = Swing.EmptyBorder(30, 30, 10, 150)
    }

    listenTo(button)
    listenTo(button2)
    listenTo(button3)

    var nClicks = 0

    // the stuff in {} is the HANDLER to the `reactions` property of the top frame
    reactions += {
      case ButtonClicked(`button`) =>
        nClicks += 1
        label.text = "Number of button clicks: " + nClicks
      case ButtonClicked(`button2`) =>
        nClicks -= 3
        label.text = "Number of button clicks is now: " + nClicks
      case ButtonClicked(`button3`) =>
        nClicks = 0
        label.text = "Reset to 0: " + nClicks
    }
  }
}
