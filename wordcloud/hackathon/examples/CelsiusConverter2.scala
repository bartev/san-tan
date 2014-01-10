package com.identified.bvoneoffs.hackathon.examples

import swing._
import swing.event.EditDone

object CelsiusConverter2 extends SimpleSwingApplication {
  def newField = new TextField {
    text = "0"
    columns = 5
    horizontalAlignment = Alignment.Right
  }
  val celsius = newField
  val farenheit = newField

  listenTo(farenheit, celsius)
  reactions += {
    case EditDone(`farenheit`) =>
      val f = Integer.parseInt(farenheit.text)
      val c = (f - 32) * 5 / 9
      celsius.text = c.toString
    case EditDone(`celsius`) =>
      val c = Integer.parseInt(celsius.text)
      val f = c * 9 / 5 + 32
      farenheit.text = f.toString
  }

  lazy val ui = new FlowPanel(celsius,
                              new Label(" Celsius = "),
                              farenheit,
                              new Label(" Farenheit")) {
    border = Swing.EmptyBorder(15, 10, 10, 10)
  }

  def top = new MainFrame {
    title = "Convert Celsius / Farenheit"
    contents = ui
  }
}
