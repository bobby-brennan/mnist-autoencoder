import {Component} from '@angular/core';
import {EncoderService} from '../services/encoder.service';

declare let window:any;

const PLOT_SIZE = 500;
const WINDOW_SIZE = 1;

@Component({
    selector: 'plot',
    templateUrl: './plot.pug',
    styles: [`
      .plot {
        height: ${PLOT_SIZE}px;
        width: ${PLOT_SIZE}px;
        border: 1px solid black;
        position: relative;
      }
      .click-point {
        background-color: black;
      }
      .point {
        position: absolute;
        height: 8px;
        width: 8px;
        border-radius: 4px;
      }
      .colors .point {
        margin-right: 10px;
      }
    `]
})
export class PlotComponent {
  xCoord:number;
  yCoord:number;
  x:number;
  y:number;
  pixels = null;
  images = null;
  labelColors = [
    'green',
    'red',
    'pink',
    'magenta',
    'orange',
    'yellow',
    'cyan',
    'blue',
    'brown',
    'gold',
  ]

  constructor(private encoder:EncoderService) {
    window.plot = this;
    this.encoder.getImages()
      .then(ims => this.images = ims);
  }

  onClick(evt) {
    this.xCoord = evt.offsetX;
    this.yCoord = evt.offsetY;
    if (evt.srcElement.className !== 'plot') {
      this.xCoord = evt.srcElement.offsetLeft;
      this.yCoord = evt.srcElement.offsetTop;
    }
    this.x = this.castCoord(this.xCoord);
    this.y = this.castCoord(this.yCoord);
    this.pixels = null;
    this.decode();
  }

  castCoord(coord) {
    return (coord / PLOT_SIZE) * WINDOW_SIZE * 2 - WINDOW_SIZE;
  }

  getCoord(val) {
    return ((val + WINDOW_SIZE) / (2 * WINDOW_SIZE)) * PLOT_SIZE;
  }

  decode() {
    this.encoder.decode([this.x, this.y])
      .then(pixels => this.pixels = pixels);
  }
}
