package com.identified.bvoneoffs.hackathon.examples

import swing._
import com.sun.java.swing.action.AlignCenterAction

object FirstSwingApp extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "Bartev's frame"
    contents = new FlowPanel{
      contents += new Button("my button")
    }
  }
}
