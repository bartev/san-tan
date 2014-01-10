package com.identified.bvoneoffs.hackathon.examples

import java.awt.Dimension
import swing._

object ColorChooserDemo extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "Color Chooser Demo"
    size = new Dimension(400, 400)
    contents = ui
  }

  def ui = new BorderPanel {
    // TODO - Can't find ColorChooser

    //    val colorChooser = new ColorChooser{
    //      reactions += {
    //        case ColorChanged(_, c) => banner.foreground = c
    //      }
    //    }
  }
}
