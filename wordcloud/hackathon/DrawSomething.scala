package com.identified.bvoneoffs.hackathon

import java.awt.geom._
import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Graphics2D, Font}
import java.io.File

object DrawSomething {
  //  http://otfried-cheong.appspot.com/scala/drawing.html
  // Use java.awt.image.Buffered image
  // obtain the java.awt.Graphics2d object for drawing this image
  // Save the image using javax.imageio.ImageIO.write

  def setupCanvas: BufferedImage = {
    // Size of image
    val size = (500, 500)

    // Create an image
    val canvas = new BufferedImage(size._1, size._2, BufferedImage.TYPE_INT_RGB)
    canvas
  }

  def setupGraphics(c: BufferedImage): Graphics2D = {
    // get Graphics2D for the image
    val g: Graphics2D = c.createGraphics()
    // clear background
    g.setColor(Color.WHITE)
    g.fillRect(0, 0, c.getWidth, c.getHeight)

    // enable anti-aliased rendering (prettier lines and circles)
    g.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING,
                       java.awt.RenderingHints.VALUE_ANTIALIAS_ON)
    g
  }
  def drawTwoFilledCircles(g: Graphics2D) {
    g.setColor(Color.RED)
    g.fill(new Ellipse2D.Double(30.0, 30.0, 40.0, 40.0))
    g.fill(new Ellipse2D.Double(230.0, 380.0, 40.0, 40.0))
  }
  def drawUnfilledCirclePenWidth3(g: Graphics2D) {
    g.setColor(Color.MAGENTA)
    g.setStroke(new BasicStroke(3f))
    g.draw(new Ellipse2D.Double(400.0, 35.0, 30.0, 30.0))
  }
  def drawFilledUnfilledRectangle(g: Graphics2D) {
    g.setColor(Color.CYAN)
    g.fill(new Rectangle2D.Double(20.0, 400.0, 50.0, 20.0))
    g.draw(new Rectangle2D.Double(400.0, 400.0, 50.0, 20.0))
  }
  def drawLine(g: Graphics2D) {
    g.setStroke(new BasicStroke()) // reset to default
    g.setColor(new Color(0, 0, 255)) // same as Color.BLUE
    g.draw(new Line2D.Double(50.0, 50.0, 250.0, 400.0))
  }
  def drawText(g: Graphics2D) {
    g.setColor(new Color(0, 128, 0)) // a darker green
    g.setFont(new Font("Batang", Font.PLAIN, 20))
    g.drawString("Hello World", 155, 225)
    g.drawString("Goodbye fish", 175, 245)
  }
  def cleanupGraphics(g: Graphics2D) {
    g.dispose()
  }

  def writeImageToFile(canvas: BufferedImage, fname: String) = {
    javax.imageio.ImageIO.write(canvas, "png", new File(fname))
  }

  def main(args: Array[String]) {
    val canvas = setupCanvas
    val g = setupGraphics(canvas)
    drawTwoFilledCircles(g)
    drawUnfilledCirclePenWidth3(g)
    drawFilledUnfilledRectangle(g)
    drawLine(g)
    drawText(g)
    cleanupGraphics(g)
    writeImageToFile(canvas, "drawing.png")

  }
}
