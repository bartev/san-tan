package com.identified.bvoneoffs.hackathon.examples

import swing.SimpleSwingApplication
import swing._
import swing.event._

object ButtonApp extends SimpleSwingApplication{
  def top = new MainFrame {
    title = "My Frame"
    contents = new GridPanel(4,1) {
      hGap = 3
      vGap = 3
      contents += new Button{
        text = "Press Me!"
        reactions += {
          case ButtonClicked(_) => text = "Hello Scala"
        }
      }
    }
    size = new Dimension(300, 80)
  }

}
