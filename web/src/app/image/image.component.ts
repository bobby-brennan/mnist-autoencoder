import {Component, Input} from '@angular/core';

@Component({
    selector: 'image',
    templateUrl: './image.pug',
    styles: [`
      .pixel-row {
        display: block;
        height: 5px;
      }
      .pixel {
        display: inline-block;
        background-color: black;
        height: 5px;
        width: 5px;
      }
    `]
})
export class ImageComponent {
  @Input() pixels;
  constructor() {}
}
