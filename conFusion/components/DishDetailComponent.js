import React, { Component } from "react";

import {  Button,Card, CardImg, CardImgOverlay, CardText, CardTitle } from "reactstrap";
import { COMMENTS } from "../shared/comments"; 

class DishDetail extends Component {
  render() {


    function RenderComments(comments,target) {
     console.log(comments,target);
      const Comments =  comments.map((comment) => {

        if ((target) === (comment.dishId)){
           return (
          <li key={comment.id}>
            <h3>About dish #:{comment.dishId}:{comment.target}
            </h3>
            <p>{comment.comment}</p>
            <p>
              {comment.author}
              {comment.date}
            </p>
          </li>
        );
        }

       
      });
      return Comments;
    }

    function RenderDish(dish) { 
      return ( 
        <div key={dish.id} className='col-12 col-md-5 m-0'>
          <Card>
            {" "}
            <CardTitle>{dish.name}</CardTitle>
            <CardImg width='100%' src={dish.image} alt={dish.name} />
            <CardImgOverlay>
              <CardText>{dish.description} </CardText>
            </CardImgOverlay>
          </Card>
        </div>
      );
    }



    if (this.props.props != null && ) {
      const dishId = this.props.props.dishId;
      return ( 
          console.log(COMMENTS,dishId),
        <div key={this.props.props.dishId} className='container row'>
          {RenderDish(this.props.props)}
          <div className='col-12 col-md-5 m-0'>
            <h4>Comments</h4>
            <ul className={"list-unstyled"}>

             {RenderComments(COMMENTS,(this.props.props.dishId))}
            </ul>
          </div>
        </div>
      );
    } else return <div><Button >comments?</Button></div>;
  }
}

export default DishDetail;
