package com.identified.bvoneoffs.hackathon.examples

import swing._
import event.{EditDone, ButtonClicked}

object CelsiusConverter extends SimpleSwingApplication{
  def top = new MainFrame{
    title = "Convert Celsius to Farenheit"
    val tempCelsius = new TextField
    val celsiusLabel = new Label{
      text = "Celsius"
      border = Swing.EmptyBorder(5, 5, 5, 5)
    }
    val convertButton = new Button{
      text = "Convert"
    }
    val farenheitLabel = new Label{
      text = "Farenheit    "
      border = Swing.EmptyBorder(5, 5, 5, 5)
      listenTo(convertButton, tempCelsius)

      def convert() {
        val c = Integer.parseInt(tempCelsius.text)
        val f = c * 9/5 + 32
        text = "<html><font color = red>" + f + "</font> yippee </html>"
      }

      reactions += {
        case ButtonClicked(_) | EditDone(_) => convert()
      }
    }
    contents = new GridPanel(2, 2) {
      contents.append(tempCelsius, celsiusLabel, convertButton, farenheitLabel)
      border = Swing.EmptyBorder(10, 10, 10, 10)
    }
  }
}
