import {Injectable} from '@angular/core'
import { Http, Headers } from '@angular/http';

const HOST = "http://54.203.20.214:3000"

@Injectable()
export class EncoderService {
  constructor(private http:Http) {}

  post(path, data) {
    let req = {
      headers: new Headers(),
    }
    req.headers.append('Content-Type', 'application/json');
    return this.http.post(HOST + path, JSON.stringify(data), req)
        .toPromise()
        .then(data => data.json())
  }

  encode(image) {
    return this.post('/encode', image);
  }

  decode(encoding) {
    return this.post('/decode', encoding);
  }

  getImages() {
    return this.http.get(HOST + '/images')
        .toPromise()
        .then(data => data.json())
  }
}
